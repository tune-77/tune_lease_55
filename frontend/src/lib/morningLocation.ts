import { formatLocalDateKey } from "@/lib/date";

// REV-422: その日最初のチャット時の現在地を1日固定で使う（紫苑に渡す天気の地域決定用）。
// 約10km単位へ丸め、ブラウザ内にだけ保存する。拒否・失敗時もその日は再取得しない。
const STORAGE_KEY = "shion-morning-location";

type StoredLocation = { date: string; lat: number | null; lon: number | null };
export type WeatherLocation = { weather_lat?: number; weather_lon?: number };

const toPayload = (loc: StoredLocation): WeatherLocation =>
  loc.lat === null || loc.lon === null ? {} : { weather_lat: loc.lat, weather_lon: loc.lon };

const save = (loc: StoredLocation) => {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(loc));
  } catch {
    // localStorage 不可でも送信は続ける
  }
};

export const getMorningLocation = async (): Promise<WeatherLocation> => {
  if (typeof window === "undefined") return {};
  const today = formatLocalDateKey();
  try {
    const cached = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "null") as StoredLocation | null;
    if (cached?.date === today) return toPayload(cached);
  } catch {
    // 壊れた値は取り直す
  }
  const loc = await new Promise<StoredLocation>((resolve) => {
    if (!navigator.geolocation) {
      resolve({ date: today, lat: null, lon: null });
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) =>
        resolve({
          date: today,
          lat: Math.round(pos.coords.latitude * 10) / 10,
          lon: Math.round(pos.coords.longitude * 10) / 10,
        }),
      () => resolve({ date: today, lat: null, lon: null }),
      { timeout: 5000, maximumAge: 30 * 60 * 1000, enableHighAccuracy: false },
    );
  });
  save(loc);
  return toPayload(loc);
};
