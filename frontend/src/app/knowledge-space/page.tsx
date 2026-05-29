"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RefreshCw, Search, Sparkles, Network, FileText, SlidersHorizontal, X } from "lucide-react";

type GraphNode = {
  id: string;
  label: string;
  type: "cluster" | "note" | "external";
  category: string;
  color: string;
  radius?: number;
  path?: string;
  chunk_count?: number;
  link_count?: number;
  mtime?: number;
  sections?: string[];
  count?: number;
};

type GraphEdge = {
  source: string;
  target: string;
  type: string;
  weight?: number;
  color?: string;
  mtime?: number;
  recent_rank?: number;
};

type KnowledgeGraph = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  summary: {
    indexed_chunks?: number;
    notes?: number;
    shown_nodes?: number;
    links?: number;
    limit?: number;
  };
  legend: Array<{ label: string; category: string; color: string }>;
};

type SceneMode = "all" | "recent" | "search" | "evidence";
type VisualMode = "practical" | "galaxy";

const hashValue = (text: string) => {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash >>> 0);
};

const buildPositions = (nodes: GraphNode[]) => {
  const positions = new Map<string, THREE.Vector3>();
  const clusterNodes = nodes.filter((node) => node.type === "cluster");
  const categoryOrder = ["asset", "research", "knowledge", "case", "feedback", "wiki", "daily", "external"];
  const noteGroups = new Map<string, GraphNode[]>();
  nodes.filter((node) => node.type !== "cluster").forEach((node) => {
    const category = node.type === "external" ? "external" : node.category;
    const arr = noteGroups.get(category) || [];
    arr.push(node);
    noteGroups.set(category, arr);
  });

  clusterNodes.forEach((node) => {
    const category = node.id.replace("cluster:", "");
    const arm = Math.max(0, categoryOrder.indexOf(category));
    const angle = (arm / Math.max(1, categoryOrder.length)) * Math.PI * 2;
    const radius = 12 + arm * 2.5;
    positions.set(node.id, new THREE.Vector3(Math.cos(angle) * radius, (arm % 2 ? 2 : -2), Math.sin(angle) * radius));
  });

  nodes.forEach((node) => {
    if (positions.has(node.id)) return;
    const seed = hashValue(node.id);
    const category = node.type === "external" ? "external" : node.category;
    const group = noteGroups.get(category) || [node];
    const index = Math.max(0, group.findIndex((item) => item.id === node.id));
    const arm = Math.max(0, categoryOrder.indexOf(category));
    const baseAngle = (arm / Math.max(1, categoryOrder.length)) * Math.PI * 2;
    const normalized = index / Math.max(1, group.length - 1);
    const radius = node.type === "external"
      ? 185 + (seed % 115)
      : 38 + Math.sqrt(index + 1) * 16 + normalized * 96;
    const spiral = baseAngle + radius * 0.047 + ((seed % 100) / 100 - 0.5) * 0.30;
    const jitter = ((seed >> 8) % 100) / 100 - 0.5;
    const vertical = node.type === "external"
      ? jitter * 42
      : jitter * (5 + normalized * 12);
    const armWidth = node.type === "external" ? 24 : 4 + normalized * 10;
    const tangent = new THREE.Vector3(-Math.sin(spiral), 0, Math.cos(spiral)).multiplyScalar((((seed >> 16) % 100) / 100 - 0.5) * armWidth);
    const position = new THREE.Vector3(
      Math.cos(spiral) * radius,
      vertical,
      Math.sin(spiral) * radius,
    );
    positions.set(node.id, position.add(tangent));
  });

  return positions;
};

const buildGalaxyDust = (nodes: GraphNode[]) => {
  const positions: number[] = [];
  const colors: number[] = [];
  const categories = ["asset", "research", "knowledge", "case", "feedback", "wiki", "daily"];
  const palette = new Map(nodes.map((node) => [node.category, node.color]));
  const dustCount = Math.min(2600, Math.max(1100, nodes.length * 14));

  for (let i = 0; i < dustCount; i += 1) {
    const category = categories[i % categories.length];
    const arm = i % categories.length;
    const baseAngle = (arm / categories.length) * Math.PI * 2;
    const radius = 18 + Math.pow(i / dustCount, 0.58) * 320;
    const seed = hashValue(`${category}:${i}`);
    const angle = baseAngle + radius * 0.048 + ((seed % 100) / 100 - 0.5) * 0.34;
    const spread = (((seed >> 8) % 100) / 100 - 0.5) * (9 + radius * 0.055);
    const tangent = new THREE.Vector3(-Math.sin(angle), 0, Math.cos(angle)).multiplyScalar(spread);
    const y = (((seed >> 16) % 100) / 100 - 0.5) * (5 + radius * 0.035);
    const point = new THREE.Vector3(Math.cos(angle) * radius, y, Math.sin(angle) * radius).add(tangent);
    positions.push(point.x, point.y, point.z);
    const color = new THREE.Color(palette.get(category) || "#e2e8f0").lerp(new THREE.Color("#f8fafc"), 0.45);
    colors.push(color.r, color.g, color.b);
  }

  return { positions, colors };
};

const createStarTexture = () => {
  const size = 96;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  if (!context) return null;

  const center = size / 2;
  const gradient = context.createRadialGradient(center, center, 0, center, center, center);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.16, "rgba(255,255,255,0.95)");
  gradient.addColorStop(0.34, "rgba(255,244,210,0.42)");
  gradient.addColorStop(0.72, "rgba(125,190,255,0.11)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, size, size);

  context.strokeStyle = "rgba(255,255,255,0.72)";
  context.lineWidth = 1.2;
  context.beginPath();
  context.moveTo(center, 8);
  context.lineTo(center, size - 8);
  context.moveTo(8, center);
  context.lineTo(size - 8, center);
  context.stroke();

  context.strokeStyle = "rgba(255,239,184,0.36)";
  context.lineWidth = 0.8;
  context.beginPath();
  context.moveTo(22, 22);
  context.lineTo(size - 22, size - 22);
  context.moveTo(size - 22, 22);
  context.lineTo(22, size - 22);
  context.stroke();

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
};

function KnowledgeSpaceScene({
  graph,
  onSelect,
  selectedId,
  searchTerm,
  timePercent,
  mode,
  visualMode,
}: {
  graph: KnowledgeGraph;
  onSelect: (node: GraphNode | null) => void;
  selectedId?: string;
  searchTerm: string;
  timePercent: number;
  mode: SceneMode;
  visualMode: VisualMode;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !graph.nodes.length) return;

    const scene = new THREE.Scene();
    const isGalaxy = visualMode === "galaxy";
    scene.background = new THREE.Color(isGalaxy ? "#05070d" : "#070b12");
    scene.fog = new THREE.FogExp2(isGalaxy ? "#05070d" : "#070b12", isGalaxy ? 0.00145 : 0.0022);

    const isCompact = mount.clientWidth < 640;
    const camera = new THREE.PerspectiveCamera(58, mount.clientWidth / Math.max(1, mount.clientHeight), 0.1, 1400);
    camera.position.set(0, isCompact ? 310 : 310, isCompact ? 260 : 260);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.rotateSpeed = 0.45;
    controls.zoomSpeed = 0.75;
    controls.minDistance = 75;
    controls.maxDistance = 520;
    controls.target.set(0, 0, 0);

    scene.add(new THREE.AmbientLight("#dbeafe", isGalaxy ? 0.42 : 0.55));
    const keyLight = new THREE.DirectionalLight("#ffffff", isGalaxy ? 1.15 : 0.85);
    keyLight.position.set(80, 120, 80);
    scene.add(keyLight);
    const coreLight = new THREE.PointLight("#fef3c7", isGalaxy ? 4.5 : 1.2, 300);
    coreLight.position.set(0, 0, 0);
    scene.add(coreLight);
    const rimLight = new THREE.PointLight("#38bdf8", isGalaxy ? 1.9 : 0.8, 420);
    rimLight.position.set(-180, 30, 150);
    scene.add(rimLight);

    const root = new THREE.Group();
    root.position.y = isCompact ? 75 : 0;
    scene.add(root);

    const positions = buildPositions(graph.nodes);
    const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
    const starById = new Map<string, THREE.Sprite>();
    const rayTargets: THREE.Object3D[] = [];
    const starMaterials: THREE.SpriteMaterial[] = [];
    const flowMaterials: THREE.SpriteMaterial[] = [];
    const starTexture = createStarTexture();
    const maxLinks = Math.max(1, ...graph.nodes.map((node) => node.link_count || node.count || 0));
    const linkPower = (node: GraphNode) => Math.sqrt(Math.min(1, Math.log1p(node.link_count || node.count || 0) / Math.log1p(maxLinks)));
    const noteTimes = graph.nodes.filter((node) => node.type === "note" && node.mtime).map((node) => Number(node.mtime));
    const minTime = noteTimes.length ? Math.min(...noteTimes) : 0;
    const maxTime = noteTimes.length ? Math.max(...noteTimes) : 0;
    const cutoffTime = minTime + (maxTime - minTime) * (timePercent / 100);
    const terms = searchTerm.toLowerCase().split(/\s+/).map((term) => term.trim()).filter(Boolean);
    const matchesSearch = (node: GraphNode) => {
      if (!terms.length) return true;
      const haystack = [node.label, node.path, node.category, ...(node.sections || [])].join(" ").toLowerCase();
      return terms.every((term) => haystack.includes(term));
    };
    const activeNodeIds = new Set<string>();
    graph.nodes.forEach((node) => {
      const inTime = node.type !== "note" || !node.mtime || !maxTime || Number(node.mtime) <= cutoffTime;
      const matched = matchesSearch(node);
      const selected = selectedId && node.id === selectedId;
      const connected = selectedId && graph.edges.some((edge) => (
        (edge.source === selectedId && edge.target === node.id) || (edge.target === selectedId && edge.source === node.id)
      ));
      if ((mode === "recent" && inTime) || (mode === "search" && matched) || (mode === "evidence" && matched) || mode === "all" || selected || connected) {
        activeNodeIds.add(node.id);
      }
    });

    const dust = isGalaxy ? buildGalaxyDust(graph.nodes) : { positions: [], colors: [] };
    const dustGeometry = new THREE.BufferGeometry();
    dustGeometry.setAttribute("position", new THREE.Float32BufferAttribute(dust.positions, 3));
    dustGeometry.setAttribute("color", new THREE.Float32BufferAttribute(dust.colors, 3));
    const dustMaterial = new THREE.PointsMaterial({
      size: 1.05,
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      opacity: isGalaxy ? 0.46 : 0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    root.add(new THREE.Points(dustGeometry, dustMaterial));

    const coreGeometry = new THREE.SphereGeometry(10, 40, 24);
    const coreMaterial = new THREE.MeshBasicMaterial({
      color: "#f8fafc",
      transparent: true,
      opacity: isGalaxy ? 0.72 : 0,
      blending: THREE.AdditiveBlending,
    });
    const core = new THREE.Mesh(coreGeometry, coreMaterial);
    root.add(core);

    const linePositions: number[] = [];
    const lineColors: number[] = [];
    graph.edges.forEach((edge) => {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) return;
      linePositions.push(source.x, source.y, source.z, target.x, target.y, target.z);
      const color = new THREE.Color(edge.color || (edge.type === "wikilink" ? "#38bdf8" : "#64748b"))
        .lerp(new THREE.Color("#e0f2fe"), edge.type === "wikilink" ? 0.58 : 0.42);
      lineColors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    });
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
    lineGeometry.setAttribute("color", new THREE.Float32BufferAttribute(lineColors, 3));
    const lineMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: isGalaxy ? 0.1 : 0.24,
      blending: isGalaxy ? THREE.AdditiveBlending : THREE.NormalBlending,
      depthWrite: false,
    });
    root.add(new THREE.LineSegments(lineGeometry, lineMaterial));

    const selectedRoutes = selectedId
      ? graph.edges
        .filter((edge) => edge.source === selectedId || edge.target === selectedId)
        .filter((edge) => positions.has(edge.source) && positions.has(edge.target))
        .slice(0, isGalaxy ? 14 : 8)
        .map((edge, index) => ({
          source: positions.get(edge.source)!,
          target: positions.get(edge.target)!,
          offset: index * 0.07,
          speed: 0.28 + (index % 4) * 0.035,
        }))
      : [];

    const recentRoutes = graph.edges
      .slice()
      .sort((a, b) => (a.recent_rank || 9999) - (b.recent_rank || 9999))
      .filter((edge) => positions.has(edge.source) && positions.has(edge.target))
      .slice(0, isGalaxy ? 5 : 0)
      .map((edge, index) => ({
        source: positions.get(edge.source)!.clone(),
        target: positions.get(edge.target)!.clone(),
        offset: index * 0.19,
        speed: 0.12 + index * 0.012,
      }));

    const routeGlowPositions: number[] = [];
    const routeGlowColors: number[] = [];
    [...recentRoutes, ...selectedRoutes].forEach((route, index) => {
      routeGlowPositions.push(route.source.x, route.source.y, route.source.z, route.target.x, route.target.y, route.target.z);
      const color = new THREE.Color(index < recentRoutes.length ? "#dff7ff" : "#fef3c7");
      routeGlowColors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    });
    const routeGlowGeometry = new THREE.BufferGeometry();
    routeGlowGeometry.setAttribute("position", new THREE.Float32BufferAttribute(routeGlowPositions, 3));
    routeGlowGeometry.setAttribute("color", new THREE.Float32BufferAttribute(routeGlowColors, 3));
    const routeGlowMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: isGalaxy ? 0.24 : 0.16,
      blending: isGalaxy ? THREE.AdditiveBlending : THREE.NormalBlending,
      depthWrite: false,
    });
    root.add(new THREE.LineSegments(routeGlowGeometry, routeGlowMaterial));

    const animatedRoutes = [...recentRoutes, ...selectedRoutes];
    const flowLights = animatedRoutes.flatMap((route, routeIndex) => {
      return (isGalaxy ? [0, 1, 2] : [0]).map((trailIndex) => {
        const material = new THREE.SpriteMaterial({
          map: starTexture || undefined,
          color: trailIndex === 0 ? "#ffffff" : trailIndex === 1 ? "#bff4ff" : "#7dd3fc",
          transparent: true,
          opacity: isGalaxy ? (trailIndex === 0 ? 0.98 : trailIndex === 1 ? 0.48 : 0.2) : 0.72,
          depthWrite: false,
          blending: isGalaxy ? THREE.AdditiveBlending : THREE.NormalBlending,
        });
        const sprite = new THREE.Sprite(material);
        sprite.scale.setScalar(isGalaxy ? (trailIndex === 0 ? 14 : trailIndex === 1 ? 9 : 5.5) : 8);
        sprite.userData.routeIndex = routeIndex;
        sprite.userData.trailIndex = trailIndex;
        root.add(sprite);
        flowMaterials.push(material);
        return sprite;
      });
    });

    graph.nodes.forEach((node) => {
      const position = positions.get(node.id);
      if (!position) return;
      const power = linkPower(node);
      const active = activeNodeIds.has(node.id);
      const inTime = node.type !== "note" || !node.mtime || !maxTime || Number(node.mtime) <= cutoffTime;
      const matched = matchesSearch(node);
      const emphasis = (mode === "search" || mode === "evidence") && matched ? 1.35 : selectedId === node.id ? 1.45 : 1;
      const dim = (mode === "recent" && !inTime) || ((mode === "search" || mode === "evidence") && terms.length > 0 && !matched) ? 0.22 : active ? 1 : 0.55;
      const baseSize = isGalaxy
        ? (node.type === "cluster" ? 16 : node.type === "external" ? 6.5 : 8.5)
        : (node.type === "cluster" ? 11 : node.type === "external" ? 4 : 5.5);
      const starSize = (baseSize + power * (node.type === "cluster" ? (isGalaxy ? 28 : 12) : (isGalaxy ? 21 : 9))) * emphasis;
      const color = new THREE.Color(node.color || "#e2e8f0").lerp(new THREE.Color("#fff7d6"), isGalaxy ? 0.25 + power * 0.38 : 0.08 + power * 0.12);
      const material = new THREE.SpriteMaterial({
        map: starTexture || undefined,
        color,
        transparent: true,
        opacity: (isGalaxy ? 0.48 + power * 0.48 : 0.62 + power * 0.22) * dim,
        depthWrite: false,
        blending: isGalaxy ? THREE.AdditiveBlending : THREE.NormalBlending,
      });
      const star = new THREE.Sprite(material);
      star.position.copy(position);
      star.scale.setScalar(starSize);
      star.userData.nodeId = node.id;
      star.userData.visualSize = starSize;
      root.add(star);
      starById.set(node.id, star);
      starMaterials.push(material);
      rayTargets.push(star);

      if (isGalaxy ? (power > 0.42 || emphasis > 1) : emphasis > 1) {
        const flareMaterial = new THREE.SpriteMaterial({
          map: starTexture || undefined,
          color: color.clone().lerp(new THREE.Color("#ffffff"), 0.38),
          transparent: true,
          opacity: (isGalaxy ? 0.12 + power * 0.18 + (emphasis > 1 ? 0.18 : 0) : 0.22) * dim,
          depthWrite: false,
          blending: isGalaxy ? THREE.AdditiveBlending : THREE.NormalBlending,
        });
        const flare = new THREE.Sprite(flareMaterial);
        flare.position.copy(position);
        flare.scale.setScalar(starSize * (isGalaxy ? 1.9 + power * 1.1 + (emphasis > 1 ? 0.7 : 0) : 1.45));
        root.add(flare);
        starMaterials.push(flareMaterial);
      }
    });

    const haloGeometry = new THREE.RingGeometry(10, 10.7, 64);
    const haloMaterial = new THREE.MeshBasicMaterial({ color: "#f8fafc", transparent: true, opacity: 0.0, side: THREE.DoubleSide });
    const halo = new THREE.Mesh(haloGeometry, haloMaterial);
    scene.add(halo);

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let hoveredId = "";

    const updatePointer = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(rayTargets, false)[0];
      hoveredId = hit?.object?.userData?.nodeId || "";
      renderer.domElement.style.cursor = hoveredId ? "pointer" : "grab";
      const star = hoveredId ? starById.get(hoveredId) : null;
      if (star) {
        halo.visible = true;
        halo.position.copy(star.position);
        halo.quaternion.copy(camera.quaternion);
        const node = nodeById.get(hoveredId);
        const r = Math.max(14, star.userData.visualSize || (node?.radius || 5) + 10);
        halo.scale.setScalar(r / 13);
        (halo.material as THREE.MeshBasicMaterial).opacity = 0.55;
        const tip = tooltipRef.current;
        if (tip && node) {
          const stem = node.path ? node.path.replace(/^.*\//, "").replace(/\.md$/i, "") : null;
          const showStem = stem && stem !== node.label;
          tip.textContent = "";
          const nameEl = document.createElement("span");
          nameEl.style.cssText = "font-weight:700;color:#e2e8f0";
          nameEl.textContent = node.label;
          tip.appendChild(nameEl);
          if (showStem && stem) {
            const br1 = document.createElement("br");
            const stemEl = document.createElement("span");
            stemEl.style.cssText = "font-size:10px;color:#94a3b8";
            stemEl.textContent = stem;
            tip.appendChild(br1);
            tip.appendChild(stemEl);
          }
          if (node.link_count) {
            const br2 = document.createElement("br");
            const linkEl = document.createElement("span");
            linkEl.style.cssText = "font-size:10px;color:#7dd3fc";
            linkEl.textContent = `${node.link_count} links`;
            tip.appendChild(br2);
            tip.appendChild(linkEl);
          }
          tip.style.left = `${event.clientX - rect.left + 14}px`;
          tip.style.top = `${event.clientY - rect.top - 44}px`;
          tip.style.display = "block";
        }
      } else {
        (halo.material as THREE.MeshBasicMaterial).opacity = 0;
        const tip = tooltipRef.current;
        if (tip) tip.style.display = "none";
      }
    };

    const clickNode = () => {
      onSelect(hoveredId ? nodeById.get(hoveredId) || null : null);
    };

    renderer.domElement.addEventListener("pointermove", updatePointer);
    renderer.domElement.addEventListener("click", clickNode);

    const resize = () => {
      if (!mount) return;
      const width = mount.clientWidth;
      const height = Math.max(360, mount.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener("resize", resize);

    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      root.rotation.y += isGalaxy ? 0.0012 : 0.00025;
      root.rotation.x = (isCompact ? -0.64 : -0.62) + Math.sin(Date.now() * 0.00016) * 0.02;
      const elapsed = performance.now() * 0.001;
      flowLights.forEach((sprite) => {
        const route = animatedRoutes[sprite.userData.routeIndex as number];
        if (!route) return;
        const trailIndex = sprite.userData.trailIndex as number;
        const progress = (elapsed * route.speed + route.offset - trailIndex * 0.035) % 1;
        sprite.position.lerpVectors(route.source, route.target, progress < 0 ? progress + 1 : progress);
        const pulse = 0.68 + Math.sin((elapsed * 6.4 + route.offset * 12) - trailIndex * 0.7) * 0.18;
        sprite.scale.setScalar((isGalaxy ? (trailIndex === 0 ? 14 : trailIndex === 1 ? 9 : 5.5) : 8) * pulse);
      });
      controls.update();
      if (halo.visible) {
        halo.quaternion.copy(camera.quaternion);
      }
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      renderer.domElement.removeEventListener("pointermove", updatePointer);
      renderer.domElement.removeEventListener("click", clickNode);
      controls.dispose();
      lineGeometry.dispose();
      lineMaterial.dispose();
      routeGlowGeometry.dispose();
      routeGlowMaterial.dispose();
      dustGeometry.dispose();
      dustMaterial.dispose();
      coreGeometry.dispose();
      coreMaterial.dispose();
      haloGeometry.dispose();
      haloMaterial.dispose();
      starMaterials.forEach((material) => material.dispose());
      flowMaterials.forEach((material) => material.dispose());
      starTexture?.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [graph, onSelect, selectedId, searchTerm, timePercent, mode, visualMode]);

  return (
    <div className="absolute inset-0">
      <div ref={mountRef} className="absolute inset-0" />
      <div
        ref={tooltipRef}
        className="pointer-events-none absolute z-50 rounded-md border border-white/20 bg-slate-900/90 px-2.5 py-1.5 text-sm shadow-xl backdrop-blur-sm"
        style={{ display: "none" }}
      />
    </div>
  );
}

export default function KnowledgeSpacePage() {
  const [graph, setGraph] = useState<KnowledgeGraph | null>(null);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [limit, setLimit] = useState(180);
  const [searchTerm, setSearchTerm] = useState("");
  const [timePercent, setTimePercent] = useState(100);
  const [mode, setMode] = useState<SceneMode>("all");
  const [visualMode, setVisualMode] = useState<VisualMode>("practical");
  const [evidenceQuery, setEvidenceQuery] = useState("");
  const [showControls, setShowControls] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const fetchGraph = async (nextLimit = limit) => {
    setLoading(true);
    setError("");
    try {
      const res = await axios.get<KnowledgeGraph>("/api/knowledge/graph", { params: { limit: nextLimit } });
      setGraph(res.data);
      setSelected(null);
    } catch (err) {
      console.error(err);
      setError("現在ナレッジ機能を準備中です。しばらくしてから更新してください。");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGraph(180);
    const params = new URLSearchParams(window.location.search);
    const focus = params.get("focus") || params.get("q") || "";
    const savedEvidence = window.localStorage.getItem("knowledge-space-evidence") || "";
    const nextEvidence = focus || savedEvidence;
    if (nextEvidence) {
      setEvidenceQuery(nextEvidence);
      setSearchTerm(nextEvidence);
      setMode("evidence");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const visibleLegend = useMemo(() => graph?.legend?.filter((item) => graph.nodes.some((node) => node.category === item.category)) || [], [graph]);
  const noteTimes = useMemo(() => graph?.nodes.filter((node) => node.type === "note" && node.mtime).map((node) => Number(node.mtime)) || [], [graph]);
  const latestLabel = useMemo(() => {
    if (!noteTimes.length) return "";
    const min = Math.min(...noteTimes);
    const max = Math.max(...noteTimes);
    const cutoff = min + (max - min) * (timePercent / 100);
    return new Date(cutoff * 1000).toLocaleDateString("ja-JP");
  }, [noteTimes, timePercent]);
  const topStars = useMemo(() => (
    graph?.nodes
      .filter((node) => node.type === "note")
      .slice()
      .sort((a, b) => (b.link_count || 0) - (a.link_count || 0))
      .slice(0, 3) || []
  ), [graph]);

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#05070d] text-slate-100">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(56,189,248,0.08)_0%,rgba(15,23,42,0.18)_34%,rgba(5,7,13,0)_72%)]" />

      <div className="absolute left-0 right-0 top-0 z-30 border-b border-white/10 bg-slate-950/72 px-3 py-1.5 backdrop-blur-md md:px-4 md:py-3">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-2 md:flex-wrap md:gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-sm font-black text-cyan-200">
              <Sparkles className="h-4 w-4" />
              <span className="md:hidden">Knowledge Space</span>
              <span className="hidden md:inline">Obsidian Knowledge Space</span>
            </div>
            <h1 className="mt-1 hidden text-xl font-black tracking-normal text-white sm:block">インデックス後のつながりを3Dで見る</h1>
          </div>

          <div className="flex items-center gap-2 md:hidden">
            <button
              onClick={() => setShowControls((prev) => !prev)}
              aria-label="操作を開く"
              className="flex h-9 w-9 items-center justify-center rounded-md border border-cyan-200/20 bg-cyan-300/10 text-cyan-100"
            >
              <SlidersHorizontal className="h-4 w-4" />
            </button>
            <button
              onClick={() => setShowDetails((prev) => !prev)}
              aria-label="詳細を開く"
              className="flex h-9 w-9 items-center justify-center rounded-md border border-white/10 bg-white/5 text-slate-100"
            >
              <FileText className="h-4 w-4" />
            </button>
          </div>

          <div className="hidden items-center gap-2 md:flex">
            <label className="flex h-10 items-center gap-2 rounded-md border border-white/10 bg-white/5 px-3 text-xs font-bold text-slate-300">
              <Search className="h-4 w-4 text-cyan-200" />
              <input
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  setMode(e.target.value.trim() ? "search" : "all");
                }}
                placeholder="星座検索"
                className="w-32 bg-transparent text-sm text-white outline-none placeholder:text-slate-500 md:w-52"
              />
            </label>
            <button
              onClick={() => setVisualMode((prev) => prev === "practical" ? "galaxy" : "practical")}
              className={`h-10 rounded-md border px-3 text-xs font-black transition ${
                visualMode === "galaxy"
                  ? "border-cyan-200/30 bg-cyan-300/16 text-cyan-100"
                  : "border-emerald-200/25 bg-emerald-300/12 text-emerald-100"
              }`}
            >
              {visualMode === "galaxy" ? "銀河" : "実務"}
            </button>
            <button
              onClick={() => {
                const next = evidenceQuery || searchTerm;
                setSearchTerm(next);
                setMode(next.trim() ? "evidence" : "all");
              }}
              className="h-10 rounded-md border border-amber-200/20 bg-amber-300/10 px-3 text-xs font-black text-amber-100 transition hover:bg-amber-300/18"
            >
              AI根拠
            </button>
            <label className="flex items-center gap-2 rounded-md border border-white/10 bg-white/5 px-3 py-2 text-xs font-bold text-slate-300">
              <Search className="h-4 w-4 text-slate-400" />
              表示
              <select
                value={limit}
                onChange={(e) => {
                  const next = Number(e.target.value);
                  setLimit(next);
                  fetchGraph(next);
                }}
                className="bg-transparent text-white outline-none"
              >
                <option className="bg-slate-900" value={120}>120</option>
                <option className="bg-slate-900" value={180}>180</option>
                <option className="bg-slate-900" value={260}>260</option>
                <option className="bg-slate-900" value={360}>360</option>
              </select>
            </label>
            <button
              onClick={() => fetchGraph(limit)}
              disabled={loading}
              className="flex h-10 items-center gap-2 rounded-md bg-cyan-500 px-3 text-sm font-black text-slate-950 transition hover:bg-cyan-300 disabled:opacity-50"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
              更新
            </button>
          </div>
        </div>
      </div>

      <section className="absolute inset-0 pt-[45px] md:pt-[88px]">
        {graph && graph.nodes.length > 0 && (
          <KnowledgeSpaceScene
            graph={graph}
            onSelect={(node) => {
              setSelected(node);
              if (node) setShowDetails(true);
              if (node) setMode("all");
            }}
            selectedId={selected?.id}
            searchTerm={searchTerm}
            timePercent={timePercent}
            mode={mode}
            visualMode={visualMode}
          />
        )}

        {loading && (
          <div className="absolute inset-0 z-30 flex items-center justify-center bg-slate-950/70">
            <div className="rounded-md border border-cyan-300/25 bg-slate-900 px-5 py-4 text-sm font-bold text-cyan-100">
              3Dナレッジ空間を構築中...
            </div>
          </div>
        )}

        {!loading && error && (
          <div className="absolute inset-0 z-30 flex items-center justify-center px-4">
            <div className="max-w-lg rounded-md border border-rose-400/30 bg-rose-950/80 p-5 text-sm text-rose-100">{error}</div>
          </div>
        )}

        {!loading && graph && graph.nodes.length === 0 && (
          <div className="absolute inset-0 z-30 flex items-center justify-center px-4">
            <div className="max-w-lg rounded-md border border-white/10 bg-slate-900/88 p-5 text-sm text-slate-200">
              インデックス済みノートがありません。Obsidianの再インデックス後に更新してください。
            </div>
          </div>
        )}
      </section>

      <aside className={`${showControls ? "block" : "hidden"} absolute bottom-3 left-3 right-3 z-30 max-h-[44vh] overflow-y-auto rounded-md border border-white/10 bg-slate-950/86 p-3 shadow-2xl backdrop-blur-md md:bottom-4 md:left-4 md:right-auto md:block md:max-h-none md:w-[min(420px,calc(100vw-2rem))] md:overflow-visible md:p-4`}>
        <div className="mb-2 flex items-center justify-between md:hidden">
          <div className="text-xs font-black text-cyan-100">表示操作</div>
          <button
            onClick={() => setShowControls(false)}
            className="rounded-md border border-white/10 bg-white/5 p-1 text-slate-200"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="rounded-md bg-white/5 px-2 py-2">
            <div className="text-[10px] font-bold text-slate-400">Chunks</div>
            <div className="text-lg font-black text-white">{graph?.summary?.indexed_chunks ?? "-"}</div>
          </div>
          <div className="rounded-md bg-white/5 px-2 py-2">
            <div className="text-[10px] font-bold text-slate-400">Notes</div>
            <div className="text-lg font-black text-white">{graph?.summary?.notes ?? "-"}</div>
          </div>
          <div className="rounded-md bg-white/5 px-2 py-2">
            <div className="text-[10px] font-bold text-slate-400">Links</div>
            <div className="text-lg font-black text-white">{graph?.summary?.links ?? "-"}</div>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          {visibleLegend.map((item) => (
            <span key={item.category} className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-white/5 px-2 py-1 text-[11px] font-bold text-slate-200">
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
              {item.label}
            </span>
          ))}
        </div>

        <div className="mt-3 grid gap-2 border-t border-white/10 pt-3">
          <div className="flex flex-wrap gap-2">
            {(["practical", "galaxy"] as VisualMode[]).map((item) => (
              <button
                key={item}
                onClick={() => setVisualMode(item)}
                className={`rounded-md border px-2 py-1 text-[11px] font-black transition ${
                  visualMode === item
                    ? "border-emerald-200/40 bg-emerald-300/18 text-emerald-50"
                    : "border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
                }`}
              >
                {item === "practical" ? "実務表示" : "銀河演出"}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap gap-2">
            {(["all", "recent", "search", "evidence"] as SceneMode[]).map((item) => (
              <button
                key={item}
                onClick={() => setMode(item)}
                className={`rounded-md border px-2 py-1 text-[11px] font-black transition ${
                  mode === item
                    ? "border-cyan-200/40 bg-cyan-300/20 text-cyan-50"
                    : "border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
                }`}
              >
                {item === "all" ? "全体" : item === "recent" ? "時系列" : item === "search" ? "検索星座" : "AI根拠"}
              </button>
            ))}
          </div>
          <label className="grid gap-1 text-[11px] font-bold text-slate-300">
            <span className="flex justify-between">
              <span>知識形成タイムライン</span>
              <span className="text-cyan-100">{latestLabel || "-"}</span>
            </span>
            <input
              type="range"
              min={5}
              max={100}
              value={timePercent}
              onChange={(event) => {
                setTimePercent(Number(event.target.value));
                setMode("recent");
              }}
              className="accent-cyan-300"
            />
          </label>
          <div className="grid gap-1 text-[11px] font-bold text-slate-300">
            <div className="text-slate-400">恒星トップ</div>
            {topStars.map((node) => (
              <button
                key={node.id}
                onClick={() => {
                  setSelected(node);
                  setSearchTerm(node.label);
                  setMode("search");
                }}
                className="truncate rounded-md bg-white/5 px-2 py-1 text-left text-amber-100 transition hover:bg-amber-300/15"
              >
                {node.label} / {node.link_count || 0} links
              </button>
            ))}
          </div>
        </div>
      </aside>

      <aside className={`${showDetails ? "block" : "hidden"} absolute bottom-3 left-3 right-3 z-30 max-h-[38vh] overflow-y-auto rounded-md border border-white/10 bg-slate-950/88 p-3 shadow-2xl backdrop-blur-md md:bottom-4 md:left-auto md:right-4 md:block md:max-h-none md:w-[min(440px,calc(100vw-2rem))] md:overflow-visible md:p-4`}>
        <div className="mb-2 flex items-center justify-between md:hidden">
          <div className="text-xs font-black text-cyan-100">ノード詳細</div>
          <button
            onClick={() => setShowDetails(false)}
            className="rounded-md border border-white/10 bg-white/5 p-1 text-slate-200"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        {selected ? (
          <div>
            <div className="flex items-start gap-3">
              <div className="mt-1 rounded-md bg-white/8 p-2">
                {selected.type === "cluster" ? <Network className="h-5 w-5 text-cyan-200" /> : <FileText className="h-5 w-5 text-cyan-200" />}
              </div>
              <div className="min-w-0">
                <div className="break-words text-base font-black text-white">{selected.label}</div>
                {selected.path && <div className="mt-1 break-words text-xs font-bold text-slate-400">{selected.path}</div>}
              </div>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
              <div className="rounded-md bg-white/5 px-2 py-2">
                <div className="font-bold text-slate-400">種別</div>
                <div className="mt-1 font-black text-cyan-100">{selected.category}</div>
              </div>
              <div className="rounded-md bg-white/5 px-2 py-2">
                <div className="font-bold text-slate-400">Chunks</div>
                <div className="mt-1 font-black text-cyan-100">{selected.chunk_count ?? selected.count ?? "-"}</div>
              </div>
              <div className="rounded-md bg-white/5 px-2 py-2">
                <div className="font-bold text-slate-400">Links</div>
                <div className="mt-1 font-black text-cyan-100">{selected.link_count ?? "-"}</div>
              </div>
            </div>
            {selected.sections?.length ? (
              <div className="mt-3">
                <div className="mb-1 text-xs font-black text-slate-400">主な見出し</div>
                <div className="flex flex-wrap gap-1.5">
                  {selected.sections.slice(0, 6).map((section) => (
                    <span key={section} className="rounded-md bg-cyan-400/10 px-2 py-1 text-[11px] font-bold text-cyan-100">{section}</span>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        ) : (
          <div className="text-sm leading-6 text-slate-300">
            ノードをクリックすると、ノート名・パス・リンク数を確認できます。ドラッグで回転、ホイールでズームできます。
          </div>
        )}
      </aside>
    </main>
  );
}
