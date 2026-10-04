// AIの直前の問いかけから、ユーザーの返答候補を1つ作る（ルール方式）。
// 例: 「保存しますか？」→「保存してくれ」
const MAX_LEN = 30;

export function suggestReply(assistantText: string): string | null {
  const text = (assistantText || "").replace(/\\n/g, "\n").trim();
  // 末尾付近の「？」で終わる最後の文を取り出す
  const questions = text.match(/[^。！!？?\n]*[？?]/g);
  if (!questions) return null;
  const last = questions[questions.length - 1].trim();
  if (text.length - (text.lastIndexOf(last) + last.length) > 80) return null;

  // 読点・括弧より後ろを対象にする
  const clause = last.split(/[、,「」『』（）()]/).pop()?.trim() ?? "";

  // 「どちらにしますか」等の選択・疑問詞つきは候補を出さない
  if (/どちら|どれ|どう|何|なに|いつ|どこ|誰|いかが/.test(clause)) return null;

  // 〜しますか / 〜しましょうか / 〜しておきますか → 〜してくれ
  const suru = clause.match(/^(.+?)(?:しておき|し)(?:ますか|ましょうか)[？?]$/);
  if (suru && suru[1].length + 4 <= MAX_LEN) return `${suru[1]}してくれ`;

  if (/(?:ますか|ましょうか|ませんか|いいですか|よろしいですか)[？?]$/.test(clause)) {
    return "お願い";
  }
  return null;
}
