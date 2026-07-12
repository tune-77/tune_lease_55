import React from "react";

/**
 * インラインの **text** を <strong>text</strong> にレンダリングする。
 * それ以外のテキストはそのまま React.Fragment として返す。
 *
 * 使用例:
 *   <p className="whitespace-pre-wrap">{renderInline(text)}</p>
 */
export function renderInline(text: string): React.ReactNode[] {
  const parts = (text ?? "").split(/(\*\*[^*\n]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    return <React.Fragment key={i}>{part}</React.Fragment>;
  });
}
