"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  source_kind?: string;
  source_label?: string;
  source_highlight?: boolean;
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

type EvidenceRoute = {
  target: GraphNode;
  path: GraphNode[];
  edgeTypes: string[];
};

const hashValue = (text: string) => {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash >>> 0);
};

const buildObsidianPositions = (nodes: GraphNode[]) => {
  const positions = new Map<string, THREE.Vector3>();
  const categoryOrder = ["asset", "research", "knowledge", "case", "feedback", "wiki", "daily", "external"];
  const noteGroups = new Map<string, GraphNode[]>();
  nodes.filter((n) => n.type !== "cluster").forEach((n) => {
    const cat = n.type === "external" ? "external" : n.category;
    const arr = noteGroups.get(cat) || [];
    arr.push(n);
    noteGroups.set(cat, arr);
  });
  const clusterCenters = new Map<string, THREE.Vector3>();
  nodes.filter((n) => n.type === "cluster").forEach((n) => {
    const cat = n.id.replace("cluster:", "");
    const arm = Math.max(0, categoryOrder.indexOf(cat));
    const angle = (arm / Math.max(1, categoryOrder.length)) * Math.PI * 2;
    const r = 20 + arm * 2;
    const pos = new THREE.Vector3(Math.cos(angle) * r, (arm % 3 - 1) * 5, Math.sin(angle) * r);
    positions.set(n.id, pos);
    clusterCenters.set(cat, pos);
  });
  nodes.forEach((n) => {
    if (positions.has(n.id)) return;
    const seed = hashValue(n.id);
    if (n.type === "external") {
      const r = 80 + (seed % 35);
      const a = (seed % 1000) / 1000 * Math.PI * 2;
      positions.set(n.id, new THREE.Vector3(Math.cos(a) * r, ((seed >> 8) % 20) - 10, Math.sin(a) * r));
      return;
    }
    const cat = n.category;
    const arm = Math.max(0, categoryOrder.indexOf(cat));
    const baseAngle = (arm / Math.max(1, categoryOrder.length)) * Math.PI * 2;
    const center = clusterCenters.get(cat) || new THREE.Vector3();
    const group = noteGroups.get(cat) || [n];
    const idx = Math.max(0, group.findIndex((x) => x.id === n.id));
    const r = 3 + Math.sqrt(idx + 1) * 3.2;
    const a = baseAngle + ((seed % 1000) / 1000) * Math.PI * 2;
    const jy = ((seed >> 16) % 100) / 100 * 10 - 5;
    positions.set(n.id, new THREE.Vector3(center.x + Math.cos(a) * r, jy, center.z + Math.sin(a) * r));
  });
  return positions;
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

const buildEvidenceRoutes = (graph: KnowledgeGraph | null, selected: GraphNode | null): EvidenceRoute[] => {
  if (!graph || !selected) return [];
  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  const evidenceTargets = graph.nodes.filter((node) => (
    node.type === "note" && node.source_highlight && node.id !== selected.id
  ));

  if (selected.source_highlight) {
    return [{ target: selected, path: [selected], edgeTypes: [] }];
  }
  if (!evidenceTargets.length) return [];

  const targetIds = new Set(evidenceTargets.map((node) => node.id));
  const buildAdjacency = (includeClusterEdges: boolean) => {
    const adjacency = new Map<string, Array<{ next: string; type: string }>>();
    graph.edges.forEach((edge) => {
      if (!includeClusterEdges && edge.type === "cluster") return;
      if (!nodeById.has(edge.source) || !nodeById.has(edge.target)) return;
      const sourceList = adjacency.get(edge.source) || [];
      sourceList.push({ next: edge.target, type: edge.type });
      adjacency.set(edge.source, sourceList);
      const targetList = adjacency.get(edge.target) || [];
      targetList.push({ next: edge.source, type: edge.type });
      adjacency.set(edge.target, targetList);
    });
    return adjacency;
  };

  const findRoutes = (includeClusterEdges: boolean) => {
    const adjacency = buildAdjacency(includeClusterEdges);
    const queue: Array<{ id: string; path: string[]; edgeTypes: string[] }> = [{ id: selected.id, path: [selected.id], edgeTypes: [] }];
    const visited = new Set([selected.id]);
    const routes: EvidenceRoute[] = [];

    while (queue.length && routes.length < 3) {
      const current = queue.shift();
      if (!current) break;
      if (current.path.length > 6) continue;

      if (targetIds.has(current.id)) {
        const pathNodes = current.path.map((id) => nodeById.get(id)).filter(Boolean) as GraphNode[];
        const target = nodeById.get(current.id);
        if (target) routes.push({ target, path: pathNodes, edgeTypes: current.edgeTypes });
        continue;
      }

      const nextLinks = (adjacency.get(current.id) || [])
        .slice()
        .sort((a, b) => {
          const aNode = nodeById.get(a.next);
          const bNode = nodeById.get(b.next);
          return Number(Boolean(bNode?.source_highlight)) - Number(Boolean(aNode?.source_highlight));
        });
      nextLinks.forEach((link) => {
        if (visited.has(link.next)) return;
        visited.add(link.next);
        queue.push({
          id: link.next,
          path: [...current.path, link.next],
          edgeTypes: [...current.edgeTypes, link.type],
        });
      });
    }
    return routes;
  };

  const linkedRoutes = findRoutes(false);
  return linkedRoutes.length ? linkedRoutes : findRoutes(true);
};

function KnowledgeSpaceScene({
  graph,
  onSelect,
  onHover,
  selectedId,
  searchTerm,
  timePercent,
  mode,
  visualMode,
  categoryFilter,
}: {
  graph: KnowledgeGraph;
  onSelect: (node: GraphNode | null) => void;
  onHover: (node: GraphNode | null, x: number, y: number) => void;
  selectedId?: string;
  searchTerm: string;
  timePercent: number;
  mode: SceneMode;
  visualMode: VisualMode;
  categoryFilter: string | null;
}) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !graph.nodes.length) return;

    const scene = new THREE.Scene();
    const isGalaxy = visualMode === "galaxy";
    scene.background = new THREE.Color(isGalaxy ? "#05070d" : "#0d1117");
    scene.fog = new THREE.FogExp2(isGalaxy ? "#05070d" : "#0d1117", isGalaxy ? 0.00145 : 0.0018);

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

    scene.add(new THREE.AmbientLight("#dbeafe", isGalaxy ? 0.42 : 0.9));
    const keyLight = new THREE.DirectionalLight("#ffffff", isGalaxy ? 1.15 : 1.2);
    keyLight.position.set(80, 120, 80);
    scene.add(keyLight);
    const coreLight = new THREE.PointLight("#fef3c7", isGalaxy ? 4.5 : 1.8, 300);
    coreLight.position.set(0, 0, 0);
    scene.add(coreLight);
    const rimLight = new THREE.PointLight("#38bdf8", isGalaxy ? 1.9 : 1.4, 420);
    rimLight.position.set(-180, 30, 150);
    scene.add(rimLight);
    if (!isGalaxy) {
      const fillLight = new THREE.DirectionalLight("#c7d2fe", 0.6);
      fillLight.position.set(-60, -40, 100);
      scene.add(fillLight);
    }

    const root = new THREE.Group();
    root.position.y = isCompact ? 75 : 0;
    scene.add(root);

    const positions = isGalaxy ? buildPositions(graph.nodes) : buildObsidianPositions(graph.nodes);
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
      const haystack = [node.label, node.path, node.category, node.source_label, node.source_kind, ...(node.sections || [])].join(" ").toLowerCase();
      return terms.every((term) => haystack.includes(term));
    };
    const matchesEvidence = (node: GraphNode) => {
      if (node.source_highlight) return true;
      if (!terms.length) return false;
      return matchesSearch(node);
    };
    const activeNodeIds = new Set<string>();
    graph.nodes.forEach((node) => {
      const inTime = node.type !== "note" || !node.mtime || !maxTime || Number(node.mtime) <= cutoffTime;
      const matched = matchesSearch(node);
      const evidenceMatched = matchesEvidence(node);
      const selected = selectedId && node.id === selectedId;
      const connected = selectedId && graph.edges.some((edge) => (
        (edge.source === selectedId && edge.target === node.id) || (edge.target === selectedId && edge.source === node.id)
      ));
      if ((mode === "recent" && inTime) || (mode === "search" && matched) || (mode === "evidence" && evidenceMatched) || mode === "all" || selected || connected) {
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
      opacity: isGalaxy ? 0.1 : 0.42,
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
      const evidenceMatched = matchesEvidence(node);
      const emphasis = (mode === "search" && matched) || (mode === "evidence" && evidenceMatched) ? 1.35 : selectedId === node.id ? 1.45 : node.source_highlight ? 1.12 : 1;
      const categoryMatch = !categoryFilter || node.type === "cluster" || node.category === categoryFilter;
      const dim = ((mode === "recent" && !inTime) || (mode === "search" && terms.length > 0 && !matched) || (mode === "evidence" && !evidenceMatched) || !categoryMatch) ? 0.18 : active ? 1 : 0.55;

      if (!isGalaxy) {
        // ── Obsidian風 球体ノード ──────────────────────────────
        const r = node.type === "cluster"
          ? 1.6 + power * 1.4
          : node.type === "external"
            ? 0.35
            : 0.45 + power * 0.9;
        const nodeColor = new THREE.Color(node.color || "#7dd3fc");
        const geo = new THREE.SphereGeometry(r * emphasis, 14, 10);
        const mat = new THREE.MeshPhongMaterial({
          color: nodeColor,
          emissive: nodeColor.clone().multiplyScalar(0.28 + power * 0.22),
          shininess: 60,
          transparent: true,
          opacity: dim,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.copy(position);
        mesh.userData.nodeId = node.id;
        mesh.userData.visualSize = r * emphasis;
        root.add(mesh);
        starById.set(node.id, mesh as unknown as THREE.Sprite);
        starMaterials.push(mat as unknown as THREE.SpriteMaterial);
        rayTargets.push(mesh);
        // 選択/検索一致/引用元ノードにリング
        if (emphasis > 1 || node.source_highlight) {
          const ringGeo = new THREE.RingGeometry(r * 1.6, r * 1.95, 32);
          const ringMat = new THREE.MeshBasicMaterial({
            color: node.source_highlight ? new THREE.Color("#fbbf24") : nodeColor,
            transparent: true,
            opacity: (node.source_highlight ? 0.7 : 0.55) * dim,
            side: THREE.DoubleSide,
          });
          const ring = new THREE.Mesh(ringGeo, ringMat);
          ring.position.copy(position);
          ring.rotation.x = Math.PI / 2;
          root.add(ring);
          starMaterials.push(ringMat as unknown as THREE.SpriteMaterial);
        }
        if (node.source_highlight) {
          const badgeGeo = new THREE.SphereGeometry(Math.max(0.18, r * 0.38), 12, 8);
          const badgeMat = new THREE.MeshBasicMaterial({
            color: "#fbbf24",
            transparent: true,
            opacity: 0.92 * dim,
          });
          const badge = new THREE.Mesh(badgeGeo, badgeMat);
          badge.position.copy(position).add(new THREE.Vector3(r * 1.35, r * 1.35, 0));
          badge.userData.nodeId = node.id;
          root.add(badge);
          starMaterials.push(badgeMat as unknown as THREE.SpriteMaterial);
          rayTargets.push(badge);
        }
        return;
      }

      // ── 銀河モード: 既存 Sprite ────────────────────────────
      const baseSize = node.type === "cluster" ? 16 : node.type === "external" ? 6.5 : 8.5;
      const starSize = (baseSize + power * (node.type === "cluster" ? 28 : 21)) * emphasis;
      const color = new THREE.Color(node.color || "#e2e8f0").lerp(new THREE.Color("#fff7d6"), 0.25 + power * 0.38);
      const material = new THREE.SpriteMaterial({
        map: starTexture || undefined,
        color,
        transparent: true,
        opacity: (0.48 + power * 0.48) * dim,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
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
      if (power > 0.42 || emphasis > 1 || node.source_highlight) {
        const flareMaterial = new THREE.SpriteMaterial({
          map: starTexture || undefined,
          color: node.source_highlight ? "#fbbf24" : color.clone().lerp(new THREE.Color("#ffffff"), 0.38),
          transparent: true,
          opacity: (0.12 + power * 0.18 + (emphasis > 1 ? 0.18 : 0) + (node.source_highlight ? 0.1 : 0)) * dim,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        });
        const flare = new THREE.Sprite(flareMaterial);
        flare.position.copy(position);
        flare.scale.setScalar(starSize * (1.9 + power * 1.1 + (emphasis > 1 ? 0.7 : 0)));
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

    const clearHover = () => {
      hoveredId = "";
      renderer.domElement.style.cursor = "grab";
      halo.visible = false;
      (halo.material as THREE.MeshBasicMaterial).opacity = 0;
      onHover(null, 0, 0);
    };

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
        const node = nodeById.get(hoveredId);
        halo.visible = false;
        onHover(node || null, event.clientX, event.clientY);
      } else {
        clearHover();
      }
    };

    const clickNode = () => {
      onSelect(hoveredId ? nodeById.get(hoveredId) || null : null);
    };

    renderer.domElement.addEventListener("pointermove", updatePointer);
    renderer.domElement.addEventListener("pointerleave", clearHover);
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
      root.rotation.y += isGalaxy ? 0.0012 : 0.00015;
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
      renderer.domElement.removeEventListener("pointerleave", clearHover);
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
  }, [graph, onSelect, selectedId, searchTerm, timePercent, mode, visualMode, categoryFilter]);

  return <div ref={mountRef} className="absolute inset-0" />;
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
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const handleSelect = useCallback((node: GraphNode | null) => {
    setSelected(node);
    if (node) setShowDetails(true);
    if (node) setMode("all");
  }, []);
  const handleHover = useCallback((node: GraphNode | null, x: number, y: number) => {
    setHoveredNode(node);
    if (node) setHoverPos({ x, y });
  }, []);
  const [hoverPos, setHoverPos] = useState({ x: 0, y: 0 });
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);

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
  const sourceStats = useMemo(() => {
    if (!graph) return [];
    const map = new Map<string, { label: string; kind: string; count: number; highlighted: number; color: string }>();
    graph.nodes
      .filter((node) => node.type === "note" && (node.source_label || node.source_kind))
      .forEach((node) => {
        const kind = node.source_kind || node.category || "unknown";
        const label = node.source_label || kind;
        const current = map.get(kind) || { label, kind, count: 0, highlighted: 0, color: node.color || "#67e8f9" };
        current.count += 1;
        if (node.source_highlight) current.highlighted += 1;
        if (!current.color && node.color) current.color = node.color;
        map.set(kind, current);
      });
    return Array.from(map.values()).sort((a, b) => b.highlighted - a.highlighted || b.count - a.count);
  }, [graph]);
  const evidenceHighlights = useMemo(() => (
    graph?.nodes
      .filter((node) => node.type === "note" && node.source_highlight)
      .slice()
      .sort((a, b) => (b.link_count || 0) - (a.link_count || 0))
      .slice(0, 4) || []
  ), [graph]);
  const evidenceRoutes = useMemo(() => buildEvidenceRoutes(graph, selected), [graph, selected]);

  const visibleNodeCount = useMemo(() => {
    if (!graph) return 0;
    if (categoryFilter) return graph.nodes.filter(n => n.category === categoryFilter).length;
    if (mode === "evidence" && !searchTerm.trim()) {
      return graph.nodes.filter(n => n.type === "note" && n.source_highlight).length;
    }
    if (searchTerm.trim()) {
      const terms = searchTerm.toLowerCase().split(/\s+/).filter(Boolean);
      return graph.nodes.filter(n => {
        const h = [n.label, n.path, n.category, n.source_label, n.source_kind, ...(n.sections || [])].join(" ").toLowerCase();
        return terms.every(t => h.includes(t));
      }).length;
    }
    return graph.nodes.length;
  }, [graph, categoryFilter, mode, searchTerm]);

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
            onSelect={handleSelect}
            onHover={handleHover}
            selectedId={selected?.id}
            searchTerm={searchTerm}
            timePercent={timePercent}
            mode={mode}
            visualMode={visualMode}
            categoryFilter={categoryFilter}
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

      {hoveredNode && (
        <div
          className="pointer-events-none fixed z-50"
          style={{
            left: hoverPos.x > window.innerWidth - 260 ? hoverPos.x - 230 : hoverPos.x + 14,
            top: hoverPos.y > window.innerHeight - 120 ? hoverPos.y - 100 : hoverPos.y - 28,
          }}
        >
          <div
            className="flex max-w-[220px] items-center gap-2 rounded-full border border-white/15 bg-slate-900/92 px-3 py-1.5 shadow-xl backdrop-blur-sm"
            style={{ boxShadow: `0 0 0 1px ${hoveredNode.color}22, 0 12px 32px rgba(2,6,23,0.42)` }}
          >
            <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: hoveredNode.color }} />
            <div className="min-w-0">
              <div className="truncate text-[11px] font-bold text-white leading-tight">{hoveredNode.label}</div>
              {hoveredNode.source_highlight && (
                <div className="truncate text-[10px] font-black text-amber-100">引用元強調ノード</div>
              )}
              {hoveredNode.source_label && (
                <div className="truncate text-[10px] font-bold text-amber-200">{hoveredNode.source_label}</div>
              )}
            </div>
          </div>
        </div>
      )}

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
        <div className="grid grid-cols-4 gap-1.5 text-center">
          <div className="rounded-md bg-white/5 px-1 py-2">
            <div className="text-[9px] font-bold text-slate-400">Chunks</div>
            <div className="text-base font-black text-white">{graph?.summary?.indexed_chunks ?? "-"}</div>
          </div>
          <div className="rounded-md bg-white/5 px-1 py-2">
            <div className="text-[9px] font-bold text-slate-400">Notes</div>
            <div className="text-base font-black text-white">{graph?.summary?.notes ?? "-"}</div>
          </div>
          <div className="rounded-md bg-white/5 px-1 py-2">
            <div className="text-[9px] font-bold text-slate-400">Links</div>
            <div className="text-base font-black text-white">{graph?.summary?.links ?? "-"}</div>
          </div>
          <div className="rounded-md bg-cyan-400/10 px-1 py-2">
            <div className="text-[9px] font-bold text-cyan-400">表示中</div>
            <div className="text-base font-black text-cyan-200">{visibleNodeCount}</div>
          </div>
        </div>

        {/* カテゴリ凡例フィルター */}
        <div className="mt-3">
          <div className="mb-1.5 flex items-center justify-between">
            <span className="text-[10px] font-black uppercase tracking-widest text-slate-500">カテゴリ</span>
            {categoryFilter && (
              <button
                onClick={() => setCategoryFilter(null)}
                className="text-[10px] font-bold text-cyan-400 hover:text-white transition"
              >
                クリア
              </button>
            )}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {visibleLegend.map((item) => {
              const active = categoryFilter === item.category;
              return (
                <button
                  key={item.category}
                  onClick={() => setCategoryFilter(active ? null : item.category)}
                  className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] font-bold transition-all ${
                    active
                      ? "border-white/30 bg-white/15 text-white scale-105"
                      : categoryFilter
                        ? "border-white/5 bg-white/3 text-slate-500 opacity-50"
                        : "border-white/10 bg-white/5 text-slate-200 hover:bg-white/10 hover:border-white/20"
                  }`}
                >
                  <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: item.color }} />
                  {item.label}
                </button>
              );
            })}
          </div>
        </div>

        {sourceStats.length > 0 && (
          <div className="mt-3 border-t border-white/10 pt-3">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-[10px] font-black uppercase tracking-widest text-amber-200">根拠レイヤー</span>
              <button
                onClick={() => {
                  setSearchTerm("");
                  setMode("evidence");
                  setCategoryFilter(null);
                }}
                className="rounded-md border border-amber-200/15 bg-amber-300/8 px-2 py-1 text-[10px] font-black text-amber-100 transition hover:bg-amber-300/16"
              >
                強調のみ
              </button>
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              {sourceStats.slice(0, 6).map((item) => (
                <button
                  key={item.kind}
                  onClick={() => {
                    setSearchTerm(item.label);
                    setMode("evidence");
                    setCategoryFilter(null);
                  }}
                  className="rounded-md border border-amber-200/12 bg-amber-300/8 px-2 py-1.5 text-left transition hover:border-amber-200/30 hover:bg-amber-300/14"
                >
                  <div className="flex items-center gap-1.5">
                    <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: item.color || "#fbbf24" }} />
                    <span className="min-w-0 truncate text-[11px] font-black text-amber-100">{item.label}</span>
                  </div>
                  <div className="mt-0.5 text-[10px] font-bold text-slate-500">
                    {item.count} notes / 強調 {item.highlighted}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {evidenceHighlights.length > 0 && (
          <div className="mt-3 grid gap-1 border-t border-white/10 pt-3 text-[11px] font-bold text-slate-300">
            <div className="text-slate-400">主要根拠ノート</div>
            {evidenceHighlights.map((node) => (
              <button
                key={node.id}
                onClick={() => {
                  setSelected(node);
                  setSearchTerm(node.label);
                  setMode("evidence");
                  setShowDetails(true);
                }}
                className="truncate rounded-md bg-amber-300/10 px-2 py-1 text-left text-amber-100 transition hover:bg-amber-300/18"
              >
                {node.label}
              </button>
            ))}
          </div>
        )}

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
                {selected.source_highlight && (
                  <div className="mt-1 inline-flex items-center gap-1 rounded-md border border-amber-300/30 bg-amber-300/14 px-2 py-0.5 text-[11px] font-black text-amber-50">
                    <span className="h-1.5 w-1.5 rounded-full bg-amber-300" />
                    引用元として強調表示中
                  </div>
                )}
                {selected.source_label && (
                  <div className="mt-1 inline-flex items-center rounded-md border border-amber-300/20 bg-amber-300/10 px-2 py-0.5 text-[11px] font-black text-amber-100">
                    引用元: {selected.source_label}
                  </div>
                )}
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
            <div className="mt-3 border-t border-white/10 pt-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="text-xs font-black text-amber-100">根拠ルート</div>
                {evidenceRoutes.length > 0 && (
                  <button
                    onClick={() => {
                      setSearchTerm(evidenceRoutes[0].path.map((node) => node.label).join(" "));
                      setMode("evidence");
                    }}
                    className="rounded-md border border-amber-200/20 bg-amber-300/10 px-2 py-1 text-[10px] font-black text-amber-100 transition hover:bg-amber-300/18"
                  >
                    ルート強調
                  </button>
                )}
              </div>
              {evidenceRoutes.length > 0 ? (
                <div className="space-y-2">
                  {evidenceRoutes.map((route) => (
                    <div key={`${selected.id}-${route.target.id}`} className="rounded-md border border-amber-200/12 bg-amber-300/8 p-2">
                      <div className="mb-1 flex items-center gap-1.5 text-[11px] font-black text-amber-100">
                        <span className="h-2 w-2 shrink-0 rounded-full bg-amber-300" />
                        {route.target.label}
                      </div>
                      <div className="flex flex-wrap items-center gap-1.5">
                        {route.path.map((node, index) => (
                          <React.Fragment key={`${route.target.id}-${node.id}-${index}`}>
                            <button
                              onClick={() => {
                                setSelected(node);
                                setSearchTerm(node.label);
                                setMode(node.source_highlight ? "evidence" : "search");
                              }}
                              className={`max-w-[132px] truncate rounded-md px-2 py-1 text-[10px] font-bold transition ${
                                node.id === selected.id
                                  ? "bg-cyan-300/16 text-cyan-100"
                                  : node.source_highlight
                                    ? "bg-amber-300/16 text-amber-100"
                                    : "bg-white/6 text-slate-300 hover:bg-white/10"
                              }`}
                              title={node.path || node.label}
                            >
                              {node.label}
                            </button>
                            {index < route.path.length - 1 && (
                              <span className="text-[10px] font-black text-slate-500">
                                {route.edgeTypes[index] === "wikilink" ? "→" : route.edgeTypes[index] === "cluster" ? "・" : "↔"}
                              </span>
                            )}
                          </React.Fragment>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-md border border-white/10 bg-white/5 px-3 py-2 text-xs font-bold leading-5 text-slate-400">
                  このノードから強調済み根拠ノートへの明示リンクはまだありません。
                </div>
              )}
            </div>
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
