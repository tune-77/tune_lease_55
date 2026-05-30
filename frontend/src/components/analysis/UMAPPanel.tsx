"use client";

import React, { useEffect, useState, useRef } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis,
  Tooltip, ResponsiveContainer, ReferenceDot,
} from 'recharts';
import { Map, ChevronDown, ChevronUp, Info, Loader2 } from 'lucide-react';
import axios from 'axios';

type Point = { x: number; y: number; s: string };
type SimilarPoint = { x: number; y: number; status: string };

type Props = {
  score: number;
  umapX: number;
  umapY: number;
  similar?: SimilarPoint[] | null;
  compact?: boolean;
};

function getLevel(s: number): 'low' | 'mid' | 'high' {
  if (s >= 70) return 'high';
  if (s >= 40) return 'mid';
  return 'low';
}

const LEVEL_CONFIG = {
  high: { label: '正常域', badge: 'bg-emerald-100 text-emerald-700 border-emerald-200', bar: 'bg-emerald-400', hdr: 'text-slate-700', wrap: 'bg-white border-slate-200' },
  mid:  { label: '要確認', badge: 'bg-amber-100 text-amber-700 border-amber-200',   bar: 'bg-amber-400',   hdr: 'text-amber-800',  wrap: 'bg-amber-50 border-amber-200' },
  low:  { label: '異常域', badge: 'bg-rose-100 text-rose-700 border-rose-200',       bar: 'bg-rose-400',    hdr: 'text-rose-800',   wrap: 'bg-rose-50 border-rose-200' },
};

// 埋め込みデータはセッション中1回だけ取得
let _embeddingsCache: Point[] | null = null;

export default function UMAPPanel({ score, umapX, umapY, similar, compact = false }: Props) {
  const [expanded, setExpanded] = useState(!compact);
  const [wonPoints, setWonPoints] = useState<Point[]>([]);
  const [lostPoints, setLostPoints] = useState<Point[]>([]);
  const [loading, setLoading] = useState(false);
  const fetchedRef = useRef(false);

  useEffect(() => {
    if (!expanded || fetchedRef.current) return;
    fetchedRef.current = true;
    if (_embeddingsCache) {
      const won  = _embeddingsCache.filter(p => p.s === '成約');
      const lost = _embeddingsCache.filter(p => p.s === '失注');
      setWonPoints(won);
      setLostPoints(lost);
      return;
    }
    setLoading(true);
    axios.get<{ points: Point[] }>('/api/umap/embeddings')
      .then(res => {
        _embeddingsCache = res.data.points;
        setWonPoints(res.data.points.filter(p => p.s === '成約'));
        setLostPoints(res.data.points.filter(p => p.s === '失注'));
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [expanded]);

  const level = getLevel(score);
  const cfg = LEVEL_CONFIG[level];
  const pct = Math.min(100, Math.max(0, score));
  const currentPoint = [{ x: umapX, y: umapY }];

  return (
    <div className={`rounded-xl border p-4 ${cfg.wrap}`}>
      {/* ヘッダー */}
      <div className="flex items-center gap-2 flex-wrap">
        <Map className="w-5 h-5 text-slate-500 flex-shrink-0" />
        <span className={`font-black text-sm ${cfg.hdr}`}>財務分布マップ（UMAP）</span>
        <span className={`text-[10px] font-black px-2 py-0.5 rounded-full border ${cfg.badge}`}>
          {cfg.label} {score.toFixed(1)}
        </span>
        {compact && (
          <button onClick={() => setExpanded(e => !e)} className="ml-auto text-slate-400 hover:text-slate-600">
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        )}
      </div>

      {expanded && (
        <>
          {/* 説明 */}
          <div className="mt-3 flex items-start gap-2 p-2.5 bg-white/70 rounded-lg border border-slate-200">
            <Info className="w-3.5 h-3.5 text-slate-400 flex-shrink-0 mt-0.5" />
            <p className="text-[11px] text-slate-600 leading-relaxed">
              <strong>財務分布マップとは：</strong>過去1,900件（成約・失注）の財務データを2次元に圧縮した散布図です。
              <span className="text-emerald-600 font-bold">緑点＝成約</span>、
              <span className="text-rose-500 font-bold">赤点＝失注</span>、
              <span className="text-amber-500 font-bold">★＝今回の案件</span>。
              成約クラスターに近いほどスコアが高くなります。
            </p>
          </div>

          {/* スコアゲージ */}
          <div className="mt-3">
            <div className="flex items-center justify-between text-[10px] text-slate-400 mb-1 font-bold">
              <span>0 異常域</span>
              <span className="text-amber-600">40</span>
              <span className="text-emerald-600">70 正常域</span>
              <span>100</span>
            </div>
            <div className="relative h-2.5 bg-slate-100 rounded-full overflow-hidden">
              <div className="absolute inset-0 flex">
                <div className="w-[40%] bg-rose-50" />
                <div className="w-[30%] bg-amber-50" />
                <div className="w-[30%] bg-emerald-50" />
              </div>
              <div className="absolute top-0 bottom-0 left-[40%] w-px bg-amber-300" />
              <div className="absolute top-0 bottom-0 left-[70%] w-px bg-emerald-400" />
              <div className={`absolute top-0 left-0 h-full rounded-full transition-all duration-500 ${cfg.bar}`} style={{ width: `${pct}%` }} />
            </div>
            <div className="text-right mt-0.5 text-xs font-black text-slate-500">{score.toFixed(1)} / 100</div>
          </div>

          {/* 散布図 */}
          <div className="mt-3 h-64 w-full">
            {loading ? (
              <div className="flex items-center justify-center h-full text-slate-400 gap-2">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span className="text-sm">マップ読込中…</span>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                  <XAxis type="number" dataKey="x" hide domain={['auto', 'auto']} />
                  <YAxis type="number" dataKey="y" hide domain={['auto', 'auto']} />
                  <ZAxis range={[8, 8]} />
                  {/* 失注（赤）*/}
                  <Scatter data={lostPoints} fill="#f87171" opacity={0.35} isAnimationActive={false} />
                  {/* 成約（緑）*/}
                  <Scatter data={wonPoints} fill="#4ade80" opacity={0.4} isAnimationActive={false} />
                  {/* 近傍成約（濃い緑）*/}
                  {similar && similar.length > 0 && (
                    <Scatter
                      data={similar.map(s => ({ x: s.x, y: s.y }))}
                      fill="#16a34a"
                      opacity={0.9}
                      isAnimationActive={false}
                    />
                  )}
                  {/* 今回の案件（☆）*/}
                  <Scatter data={currentPoint} fill="#f59e0b" shape="star" isAnimationActive={false} />
                  <Tooltip
                    cursor={false}
                    content={() => null}
                  />
                  {/* 今回の案件のラベル */}
                  <ReferenceDot
                    x={umapX}
                    y={umapY}
                    r={10}
                    fill="#f59e0b"
                    stroke="#fff"
                    strokeWidth={2}
                    label={{ value: '今回', position: 'top', fontSize: 10, fontWeight: 700, fill: '#92400e' }}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* 近傍成約案件 */}
          {similar && similar.length > 0 && (
            <div className="mt-2">
              <p className="text-[10px] font-black text-slate-500 mb-1">
                近傍の成約案件（財務分布マップ上の近い成約例）
              </p>
              <div className="flex gap-2 flex-wrap">
                {similar.map((s, i) => (
                  <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 border border-emerald-200 font-bold">
                    類似事例 {i + 1}
                  </span>
                ))}
              </div>
            </div>
          )}

          <p className="mt-2 text-[10px] text-slate-400">
            ※ Isolation Forest（非線形異常検知）+ UMAP次元圧縮。スコアは成約案件1,151件で学習。
          </p>
        </>
      )}
    </div>
  );
}
