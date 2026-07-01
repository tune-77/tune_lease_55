import React, { useState, useEffect } from 'react';
import { focusNextScreeningNumber, isDraftNumericText, parseHumanNumberInput } from '../lib/numberInput';

interface SliderInputProps {
  label: string;
  name: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  hint?: string;
  quickValues?: Array<{ label: string; value: number }>;
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
  hint,
  quickValues,
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
    if (isDraftNumericText(str)) return;

    const parsed = parseHumanNumberInput(str);
    if (parsed !== null) {
      onChange(name, parsed);
    }
  };

  const handleFocus = (e: React.FocusEvent<HTMLInputElement>) => {
    setIsFocused(true);
    if (inputStr === '0') {
      setInputStr('');
      return;
    }
    e.currentTarget.select();
  };

  const handleBlur = () => {
    setIsFocused(false);
    const parsed = parseHumanNumberInput(inputStr);
    if (parsed === null) {
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
      {quickValues?.length ? (
        <div className="mb-3 flex flex-wrap gap-1.5">
          {quickValues.map((item) => (
            <button
              key={`${name}-${item.label}-${item.value}`}
              type="button"
              onClick={() => {
                const clamped = Math.min(Math.max(item.value, min), max);
                onChange(name, clamped);
                setInputStr(String(clamped));
              }}
              className={`rounded-lg border px-2.5 py-1 text-[11px] font-black transition-colors ${
                value === item.value
                  ? 'border-blue-300 bg-blue-50 text-blue-700'
                  : 'border-slate-200 bg-white text-slate-500 hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      ) : null}

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
            data-screening-number="true"
            value={inputStr}
            onChange={handleNumberChange}
            onFocus={handleFocus}
            onBlur={handleBlur}
            onKeyDown={(e) => {
              if (e.key !== 'Enter') return;
              e.preventDefault();
              const current = e.currentTarget;
              current.blur();
              window.setTimeout(() => focusNextScreeningNumber(current), 0);
            }}
            className="flex-1 min-w-0 text-right px-3 py-2 bg-white border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 sm:text-sm font-semibold text-slate-800"
          />
          <span className="text-slate-500 sm:text-sm font-medium whitespace-nowrap">{unit}</span>
        </div>
      </div>
      {hint && <p className="text-[10px] text-slate-400 mt-2">{hint}</p>}
    </div>
  );
}
