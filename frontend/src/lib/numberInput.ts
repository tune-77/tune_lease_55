export function normalizeNumericText(value: string): string {
  return String(value || "")
    .replace(/[０-９．，－]/g, (char) => {
      const code = char.charCodeAt(0);
      if (char === "．") return ".";
      if (char === "，") return ",";
      if (char === "－") return "-";
      return String.fromCharCode(code - 0xfee0);
    })
    .replace(/,/g, "")
    .replace(/\s+/g, "")
    .replace(/円/g, "");
}

export function parseHumanNumberInput(value: string): number | null {
  const text = normalizeNumericText(value);
  if (!text || text === "-" || text === "." || text === "-.") return null;

  const oku = text.match(/(-?\d+(?:\.\d+)?)億/);
  const man = text.match(/(-?\d+(?:\.\d+)?)万/);
  if (oku || man) {
    const okuAsMillion = oku ? Number(oku[1]) * 100 : 0;
    const manAsMillion = man ? Number(man[1]) / 100 : 0;
    const total = okuAsMillion + manAsMillion;
    return Number.isFinite(total) ? total : null;
  }

  const parsed = Number.parseFloat(text);
  return Number.isFinite(parsed) ? parsed : null;
}

export function isDraftNumericText(value: string): boolean {
  const text = normalizeNumericText(value);
  return text === "" || text === "-" || text === "." || text === "-.";
}

export function focusNextNumericInput(current: HTMLInputElement, selector: string): boolean {
  const scope = current.form || document;
  const inputs = Array.from(scope.querySelectorAll<HTMLInputElement>(selector))
    .filter((input) => !input.disabled && input.tabIndex !== -1);
  const index = inputs.indexOf(current);
  const next = inputs[index + 1];
  if (!next) return false;
  next.focus();
  next.select();
  return true;
}

export function focusNextScreeningNumber(current: HTMLInputElement): void {
  focusNextNumericInput(current, "[data-screening-number='true']");
}
