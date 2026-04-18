import React, { useState } from 'react';
import axios from 'axios';
import { FileText, Download, Loader2, Printer, ShieldCheck } from 'lucide-react';
import { ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';

interface ReportProps {
  apiResult: any;
  formData: any;
  gunshiText?: string;
}

export default function ReportGenerator({ apiResult, formData, gunshiText }: ReportProps) {
  const [report, setReport] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    if (!apiResult || !apiResult.hantei) {
      alert("先に審査を実行してください。");
      return;
    }
    
    setLoading(true);
    try {
      const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/api/report/generate`, {
        result_data: apiResult,
        inputs: formData
      });
      setReport(res.data.report_markdown);
    } catch (err) {
      console.error(err);
      alert("レポート生成に失敗しました。");
    } finally {
      setLoading(false);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  // レーダーチャート用のデータ作成
  const radarData = [
    { subject: "財務スコア", A: Math.min(100, Math.max(0, apiResult?.score_base || 0)) },
    { subject: "業績推移", A: Math.min(100, Math.max(0, 50 + (apiResult?.user_op_margin || 0)*2)) },
    { subject: "安定性", A: Math.min(100, Math.max(0, 50 + (apiResult?.user_equity_ratio || 0)*2)) },
    { subject: "業界平均比", A: Math.min(100, Math.max(0, 50 + ((apiResult?.user_op_margin || 0) - (apiResult?.bench_op_margin || 0))*2)) },
    { subject: "定性評価", A: Math.min(100, Math.max(0, (apiResult?.score_base || 0) + 10)) },
  ];

  return (
    <div className="mt-12 bg-white border-2 border-slate-200 rounded-2xl shadow-sm overflow-hidden text-slate-800">
      <div className="p-6 border-b border-slate-200 flex flex-col sm:flex-row justify-between items-center gap-4 bg-slate-50">
        <div>
          <h2 className="text-xl font-black text-[#1A1A2E] flex items-center gap-2">
            <FileText className="w-6 h-6 text-indigo-600" />
            最終審査稟議書（詳細・グラフ付き）出力
          </h2>
          <p className="text-xs font-bold text-slate-500 mt-1">印刷してそのまま稟議にかけられるフォーマットで出力します。</p>
        </div>
        
        <button 
          onClick={handleGenerate}
          disabled={loading || !apiResult}
          className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white font-bold py-3 px-6 rounded-xl shadow-md transition-colors flex items-center gap-2 shrink-0"
        >
          {loading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <FileText className="w-5 h-5" />
          )}
          {loading ? "システム生成中..." : "稟議書を作成する"}
        </button>
      </div>

      {report && (
        <div className="p-8 bg-neutral-50 print:bg-white print:p-0">
          <div className="flex justify-end mb-4 gap-2 print:hidden">
            <button 
              onClick={handlePrint}
              className="text-xs font-bold text-slate-700 bg-white border border-slate-300 shadow-sm px-4 py-2 rounded-lg hover:bg-slate-50 flex items-center gap-1"
            >
              <Printer className="w-4 h-4" /> 印刷・PDF化
            </button>
          </div>
          
          {/* 実際のレポート内容 (A4風UI) */}
          <div className="bg-white mx-auto max-w-4xl p-10 md:p-16 border rounded-xl shadow-xl print:shadow-none print:border-none">
            {/* ヘッダーブロック */}
            <div className="border-b-4 border-indigo-900 pb-4 mb-8 flex justify-between items-end">
              <div>
                <p className="text-gray-500 font-bold mb-1 text-sm tracking-widest">LEASE SCREENING REPORT</p>
                <h1 className="text-3xl font-black text-slate-900">案件審査 稟議書</h1>
              </div>
              <div className="text-right">
                <p className="font-bold text-lg">{formData.company_name || '新規案件（名称未設定）'} 御中</p>
                <p className="text-sm text-gray-500">作成日: {new Date().toLocaleDateString('ja-JP')}</p>
                <div className="mt-2 inline-flex items-center gap-1 bg-green-50 text-green-700 px-3 py-1 rounded border border-green-200 font-bold text-sm">
                  <ShieldCheck className="w-4 h-4" />
                  判定: {apiResult?.hantei || "未判定"}
                </div>
              </div>
            </div>

            {/* グラフサマリー */}
            <div className="flex flex-col md:flex-row gap-8 mb-10 items-center justify-center bg-slate-50 p-6 rounded-2xl border border-slate-100">
              <div className="flex-1 text-center">
                <h3 className="font-bold text-slate-700 mb-2 border-b pb-2 text-sm">総合評価レーダー</h3>
                <div className="h-48 w-full flex justify-center">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="#e2e8f0" />
                      <PolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 10, fontWeight: 'bold' }} />
                      <Radar name="Score" dataKey="A" stroke="#4f46e5" fill="#4f46e5" fillOpacity={0.4} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="flex-1">
                <h3 className="font-bold text-slate-700 mb-4 border-b pb-2 text-sm text-center">主要スコアデータ</h3>
                <div className="space-y-4">
                   <div className="flex justify-between items-center bg-white p-3 border rounded-lg shadow-sm">
                     <span className="text-xs font-bold text-slate-500">成約可能性スコア</span>
                     <span className="text-xl font-black text-indigo-700">{apiResult?.score_base?.toFixed(1) || 0} 点</span>
                   </div>
                   <div className="flex justify-between items-center bg-white p-3 border rounded-lg shadow-sm">
                     <span className="text-xs font-bold text-slate-500">顧客PD（デフォルト）</span>
                     <span className="text-xl font-black text-rose-600">{apiResult?.score_borrower?.toFixed(2) || 0} %</span>
                   </div>
                   <div className="flex justify-between items-center bg-white p-3 border rounded-lg shadow-sm">
                     <span className="text-xs font-bold text-slate-500">審査申請物件</span>
                     <span className="text-sm font-black text-slate-800">{formData.asset_name || "未設定"}</span>
                   </div>
                </div>
              </div>
            </div>

            {/* AI生成・統合テキスト (マークダウン) */}
            <div className="prose prose-slate prose-sm max-w-none">
              
              {/* 軍師の融資条件・コンパクト戦略文 */}
              {gunshiText && (
                <div className="mb-8 p-6 bg-indigo-50 border-l-4 border-indigo-600 rounded-r-lg shadow-sm">
                  <h3 className="font-bold text-indigo-900 border-b border-indigo-200 pb-2 mb-4 text-lg">
                    審査部への特記事項・融資条件 (AI軍師コメント)
                  </h3>
                  <div className="text-slate-800 font-medium leading-relaxed whitespace-pre-wrap">
                    {gunshiText.split('\n').map((line, i) => {
                      line = line.replace(/### (.*?)/g, '<h4 class="font-bold text-indigo-700 mt-4 mb-1 text-base">■ $1</h4>');
                      line = line.replace(/\*\*(.*?)\*\*/g, '<strong class="$1"></strong>'); // fix this to proper string if needed, better yet:
                      line = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                      return <span key={i} dangerouslySetInnerHTML={{ __html: line + '<br/>' }} />;
                    })}
                  </div>
                </div>
              )}

              <h3 className="font-bold border-l-4 border-slate-600 pl-3 mb-6 text-lg mt-8">基本審査レポート</h3>
              <div className="text-slate-800 font-medium leading-relaxed whitespace-pre-wrap bg-white rounded-lg">
                {report.split('\n').map((line, i) => {
                  if (line.startsWith('================')) return null;
                  if (line.includes('審査結果レポート')) return null;
                  line = line.replace(/【(.*?)】/g, '<br/><b class="text-indigo-800 text-base border-b border-indigo-100 block mb-2 pb-1 mt-4">■ $1</b>');
                  return (
                    <span key={i} dangerouslySetInnerHTML={{ __html: line + '<br/>' }} />
                  );
                })}
              </div>
            </div>

            {/* フッター署名欄 */}
            <div className="mt-16 pt-8 border-t-2 border-slate-200 flex justify-end gap-16 print:mt-12">
               <div className="text-center">
                 <p className="text-xs text-slate-500 mb-6">起案者</p>
                 <div className="w-24 border-b border-slate-400"></div>
               </div>
               <div className="text-center">
                 <p className="text-xs text-slate-500 mb-6">審査部承認</p>
                 <div className="w-24 border-b border-slate-400"></div>
               </div>
            </div>
            
          </div>
        </div>
      )}
    </div>
  );
}
