const PREFECTURE_ALIASES: Array<{ prefecture: string; aliases: string[] }> = [
  { prefecture: "北海道", aliases: ["北海道"] },
  { prefecture: "青森県", aliases: ["青森県", "青森"] },
  { prefecture: "岩手県", aliases: ["岩手県", "岩手"] },
  { prefecture: "宮城県", aliases: ["宮城県", "宮城"] },
  { prefecture: "秋田県", aliases: ["秋田県", "秋田"] },
  { prefecture: "山形県", aliases: ["山形県", "山形"] },
  { prefecture: "福島県", aliases: ["福島県", "福島"] },
  { prefecture: "茨城県", aliases: ["茨城県", "茨城"] },
  { prefecture: "栃木県", aliases: ["栃木県", "栃木"] },
  { prefecture: "群馬県", aliases: ["群馬県", "群馬"] },
  { prefecture: "埼玉県", aliases: ["埼玉県", "埼玉"] },
  { prefecture: "千葉県", aliases: ["千葉県", "千葉"] },
  { prefecture: "東京都", aliases: ["東京都", "東京"] },
  { prefecture: "神奈川県", aliases: ["神奈川県", "神奈川"] },
  { prefecture: "新潟県", aliases: ["新潟県", "新潟"] },
  { prefecture: "富山県", aliases: ["富山県", "富山"] },
  { prefecture: "石川県", aliases: ["石川県", "石川"] },
  { prefecture: "福井県", aliases: ["福井県", "福井"] },
  { prefecture: "山梨県", aliases: ["山梨県", "山梨"] },
  { prefecture: "長野県", aliases: ["長野県", "長野"] },
  { prefecture: "岐阜県", aliases: ["岐阜県", "岐阜"] },
  { prefecture: "静岡県", aliases: ["静岡県", "静岡"] },
  { prefecture: "愛知県", aliases: ["愛知県", "愛知"] },
  { prefecture: "三重県", aliases: ["三重県", "三重"] },
  { prefecture: "滋賀県", aliases: ["滋賀県", "滋賀"] },
  { prefecture: "京都府", aliases: ["京都府", "京都"] },
  { prefecture: "大阪府", aliases: ["大阪府", "大阪"] },
  { prefecture: "兵庫県", aliases: ["兵庫県", "兵庫"] },
  { prefecture: "奈良県", aliases: ["奈良県", "奈良"] },
  { prefecture: "和歌山県", aliases: ["和歌山県", "和歌山"] },
  { prefecture: "鳥取県", aliases: ["鳥取県", "鳥取"] },
  { prefecture: "島根県", aliases: ["島根県", "島根"] },
  { prefecture: "岡山県", aliases: ["岡山県", "岡山"] },
  { prefecture: "広島県", aliases: ["広島県", "広島"] },
  { prefecture: "山口県", aliases: ["山口県", "山口"] },
  { prefecture: "徳島県", aliases: ["徳島県", "徳島"] },
  { prefecture: "香川県", aliases: ["香川県", "香川"] },
  { prefecture: "愛媛県", aliases: ["愛媛県", "愛媛"] },
  { prefecture: "高知県", aliases: ["高知県", "高知"] },
  { prefecture: "福岡県", aliases: ["福岡県", "福岡"] },
  { prefecture: "佐賀県", aliases: ["佐賀県", "佐賀"] },
  { prefecture: "長崎県", aliases: ["長崎県", "長崎"] },
  { prefecture: "熊本県", aliases: ["熊本県", "熊本"] },
  { prefecture: "大分県", aliases: ["大分県", "大分"] },
  { prefecture: "宮崎県", aliases: ["宮崎県", "宮崎"] },
  { prefecture: "鹿児島県", aliases: ["鹿児島県", "鹿児島"] },
  { prefecture: "沖縄県", aliases: ["沖縄県", "沖縄"] },
];

export const extractPrefectureFromText = (value: string): string => {
  const text = (value || "").trim();
  if (!text) return "";
  for (const entry of PREFECTURE_ALIASES) {
    if (entry.aliases.some((alias) => text.includes(alias))) {
      return entry.prefecture;
    }
  }
  return "";
};

export const normalizePrefecture = (value: string): string => {
  return extractPrefectureFromText(value) || (value || "").trim();
};
