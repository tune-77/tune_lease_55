export type ShionEntryGreeting = {
  headline: string;
  body: string;
  placeholder: string;
};

const seasonalLine = (month: number): string => {
  if (month >= 3 && month <= 5) {
    return "春の案件は、期初の投資判断と資金繰りの温度差が出やすい時期です。";
  }
  if (month === 6) {
    return "雨の季節です。数字だけで乾かない違和感は、先に拾っておきます。";
  }
  if (month >= 7 && month <= 8) {
    return "夏場は稼働も資金も熱を持ちます。焦げつく前に、返済原資を見ます。";
  }
  if (month >= 9 && month <= 11) {
    return "秋は投資判断の輪郭が出る時期です。条件に残すべきものを一緒に絞れます。";
  }
  return "冬場は資金繰りの体温が見えやすい時期です。無理な案件は、早めに冷静に見ます。";
};

const timeGreeting = (hour: number): string => {
  if (hour < 5) return "まだ夜が深いですね。";
  if (hour < 10) return "おはようございます。";
  if (hour < 12) return "午前のうちに、判断の芯を整えましょう。";
  if (hour < 15) return "昼の判断時間です。";
  if (hour < 18) return "午後の案件を見ます。";
  if (hour < 22) return "こんばんは。";
  return "遅い時間ですね。";
};

const placeholderForHour = (hour: number): string => {
  if (hour < 5) return "無理に急がず、気になる案件だけ置いてください";
  if (hour < 10) return "今日見る案件や確認したい論点を入力";
  if (hour < 15) return "案件・業界・稟議文面などを入力";
  if (hour < 18) return "午後の気になる判断や条件を入力";
  if (hour < 22) return "今日残った違和感や案件相談を入力";
  return "明日に残したくない判断メモを入力";
};

export const buildShionEntryGreeting = (date: Date = new Date()): ShionEntryGreeting => {
  const hour = date.getHours();
  const month = date.getMonth() + 1;

  return {
    headline: timeGreeting(hour),
    body: seasonalLine(month),
    placeholder: placeholderForHour(hour),
  };
};
