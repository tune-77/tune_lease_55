export type LocalChatMessage = {
  id: number;
  user_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

const HISTORY_PREFIX = "lease-chat-local-history:";
const CLEARED_PREFIX = "lease-chat-cleared-at:";
const MAX_LOCAL_MESSAGES = 120;

const storageAvailable = () => typeof window !== "undefined" && Boolean(window.localStorage);

const historyKey = (userId: string) => `${HISTORY_PREFIX}${userId || "default"}`;
const clearedKey = (userId: string) => `${CLEARED_PREFIX}${userId || "default"}`;

const messageTime = (message: Pick<LocalChatMessage, "created_at">) => {
  const parsed = Date.parse(message.created_at || "");
  return Number.isFinite(parsed) ? parsed : 0;
};

const todayStartTime = () => {
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  return start.getTime();
};

const normalizeContent = (value: string) => String(value || "").replace(/\s+/g, " ").trim();

const messageSignature = (message: LocalChatMessage) =>
  `${message.role}:${normalizeContent(message.content).slice(0, 500)}`;

export const getChatClearedAt = (userId: string) => {
  if (!storageAvailable()) return 0;
  const parsed = Number(window.localStorage.getItem(clearedKey(userId)) || "0");
  return Number.isFinite(parsed) ? parsed : 0;
};

export const getChatHistorySinceIso = (userId: string) =>
  new Date(Math.max(todayStartTime(), getChatClearedAt(userId))).toISOString();

export const loadLocalChatHistory = (userId: string): LocalChatMessage[] => {
  if (!storageAvailable()) return [];
  try {
    const raw = window.localStorage.getItem(historyKey(userId));
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((item) => item?.role && item?.content) : [];
  } catch {
    return [];
  }
};

export const saveLocalChatHistory = (userId: string, messages: LocalChatMessage[]) => {
  if (!storageAvailable()) return;
  const clearedAt = Math.max(todayStartTime(), getChatClearedAt(userId));
  const filtered = messages
    .filter((message) => messageTime(message) > clearedAt)
    .slice(-MAX_LOCAL_MESSAGES);
  window.localStorage.setItem(historyKey(userId), JSON.stringify(filtered));
};

export const appendLocalChatMessages = (userId: string, additions: LocalChatMessage[]) => {
  if (!additions.length) return;
  saveLocalChatHistory(userId, mergeChatHistories(loadLocalChatHistory(userId), additions, userId));
};

export const mergeChatHistories = (
  serverMessages: LocalChatMessage[],
  localMessages: LocalChatMessage[],
  userId: string,
): LocalChatMessage[] => {
  const clearedAt = Math.max(todayStartTime(), getChatClearedAt(userId));
  const merged: LocalChatMessage[] = [];
  const seen = new Map<string, number[]>();
  for (const message of [...serverMessages, ...localMessages]) {
    const time = messageTime(message);
    if (!message || time <= clearedAt) continue;
    const signature = messageSignature(message);
    const duplicateTimes = seen.get(signature) || [];
    if (duplicateTimes.some((previous) => Math.abs(previous - time) < 120_000)) continue;
    seen.set(signature, [...duplicateTimes, time]);
    merged.push(message);
  }
  return merged
    .sort((a, b) => messageTime(a) - messageTime(b))
    .slice(-MAX_LOCAL_MESSAGES);
};

export const clearVisibleChatHistory = (userId: string) => {
  if (!storageAvailable()) return;
  window.localStorage.setItem(clearedKey(userId), String(Date.now()));
  window.localStorage.removeItem(historyKey(userId));
};
