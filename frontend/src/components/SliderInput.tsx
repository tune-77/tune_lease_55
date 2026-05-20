import React, { useState, useEffect } from 'react';

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
  unit = "百万円",
  onChange
}: SliderInputProps) {
  // ローカル文字列state: 入力中の中間状態（"-", "1.", "-0." など）を保持
  const [inputStr, setInputStr] = useState(value.toString());
  const [isFocused, setIsFocused] = useState(false);

  // スライダー操作など外部からvalueが変わったとき、フォーカス中でなければ同期する
  useEffect(() => {
    if (!isFocused) {
      setInputStr(value.toString());
    }
  }, [value, isFocused]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(name, Number(e.target.value));
  };

  const handleNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const str = e.target.value;
    setInputStr(str);

    // "-" や "." や "-." など入力途中の状態は親に伝えない
    if (str === '' || str === '-' || str === '.' || str === '-.') return;

    const parsed = parseFloat(str);
    if (!isNaN(parsed)) {
      onChange(name, parsed);
    }
  };

  const handleFocus = () => {
    setIsFocused(true);
  };

  const handleBlur = () => {
    setIsFocused(false);
    const parsed = parseFloat(inputStr);
    if (isNaN(parsed)) {
      // 不完全な入力はリセット
      setInputStr(value.toString());
    } else {
      // blur時にmin/maxでクランプして確定
      const clamped = Math.min(Math.max(parsed, min), max);
      onChange(name, clamped);
      setInputStr(clamped.toString());
    }
  };

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
            value={Math.min(Math.max(value, min), max)}
            onChange={handleSliderChange}
            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
        </div>

        {/* 数値入力領域: type="text" + inputMode="decimal" でスマホ数字キーボード表示 */}
        <div className="w-full sm:w-1/3 flex items-center gap-2">
          <input
            type="text"
            inputMode="decimal"
            name={`${name}_number`}
            value={inputStr}
            onChange={handleNumberChange}
            onFocus={handleFocus}
            onBlur={handleBlur}
            className="flex-1 min-w-0 text-right px-3 py-2 bg-white border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 sm:text-sm font-semibold text-slate-800"
          />
          <span className="text-slate-500 sm:text-sm font-medium whitespace-nowrap">{unit}</span>
        </div>
      </div>
    </div>
  );
}
