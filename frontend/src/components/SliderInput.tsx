import React from 'react';

interface SliderInputProps {
  label: string;
  name: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (name: string, value: number) => void;
}

export default function SliderInput({
  label,
  name,
  value,
  min,
  max,
  step,
  unit = "千円",
  onChange
}: SliderInputProps) {
  
  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(name, Number(e.target.value));
  };

  const handleNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // 空白やパース不能な値は0にフォールバック、または一旦そのままにする制御が必要ですが
    // 今回は安全のため常にNumber変換します
    let val = parseInt(e.target.value, 10);
    if (isNaN(val)) val = 0;
    // max制限は直接入力ではあえて緩めることもありますが、安全性のため設定
    onChange(name, val);
  };

  // 3桁区切りのフォーマット
  const formattedValue = new Intl.NumberFormat('ja-JP').format(value);

  return (
    <div className="mb-5 bg-slate-50/50 p-4 rounded-xl border border-slate-100">
      <div className="flex justify-between items-end mb-3">
        <label className="block text-sm font-bold text-slate-700">
          {label}
        </label>
        <span className="text-xs font-semibold text-slate-500 bg-white px-2 py-1 rounded-md shadow-sm border border-slate-200">
          採用値: <span className="text-blue-600 text-sm">{formattedValue}</span> {unit}
        </span>
      </div>
      
      <div className="flex flex-col sm:flex-row gap-4 items-center">
        {/* Slider 領域 */}
        <div className="w-full sm:w-2/3">
          <input 
            type="range"
            name={`${name}_slider`}
            min={min}
            max={max}
            step={step}
            value={Math.min(value, max)} // スライダーは最大値でおさめる
            onChange={handleSliderChange}
            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
        </div>

        {/* 数値入力領域 */}
        <div className="w-full sm:w-1/3 relative">
          <input
            type="number"
            name={`${name}_number`}
            value={value.toString()} // 先頭の0などを避ける
            onChange={handleNumberChange}
            className="w-full text-right pr-12 pl-3 py-2 bg-white border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 sm:text-sm font-semibold text-slate-800"
          />
          <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
            <span className="text-slate-400 sm:text-sm font-medium">{unit}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
