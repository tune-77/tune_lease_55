const KNOWLEDGE_SPACE_EVIDENCE_KEY = "knowledge-space-evidence";
const KNOWLEDGE_SPACE_SOURCE_KEY = "knowledge-space-source";

export const buildKnowledgeSpaceHref = (query: string) => {
  const trimmed = query.replace(/\s+/g, " ").trim().slice(0, 160);
  return trimmed ? `/knowledge-space?focus=${encodeURIComponent(trimmed)}` : "/knowledge-space";
};

export const openKnowledgeSpaceFocus = (query: string, source: string) => {
  if (typeof window === "undefined") return;
  const trimmed = query.replace(/\s+/g, " ").trim().slice(0, 160);
  if (trimmed) {
    window.localStorage.setItem(KNOWLEDGE_SPACE_EVIDENCE_KEY, trimmed);
  }
  window.localStorage.setItem(
    KNOWLEDGE_SPACE_SOURCE_KEY,
    JSON.stringify({
      source,
      query: trimmed,
      opened_at: new Date().toISOString(),
    }),
  );
  window.open(buildKnowledgeSpaceHref(trimmed), "_blank");
};
