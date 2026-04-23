export type AuditCategory = "ui" | "model" | "edit" | "history" | "system";

export type AuditEventInput = {
  category: AuditCategory;
  action: string;
  featureKey?: string;
  page?: string;
  detail?: Record<string, unknown>;
};

export type AuditQueuedEvent = AuditEventInput & {
  eventId: string;
  occurredAt: string;
};

export type AuditIdentity = {
  userId: string;
  sessionId: string;
};

export type AuditLogFn = (event: AuditEventInput) => void;

export const AUDIT_CODE_STORAGE_KEY = "gam-lab-kuerzel";
export const AUDIT_SESSION_STORAGE_KEY = "gam-lab-session-id";

export const createAuditId = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `audit-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

export const getCurrentPage = () => {
  if (typeof window === "undefined") return undefined;
  return `${window.location.pathname}${window.location.search}`;
};

export const sanitizeAuditText = (value: string | null | undefined, maxLength = 160) => {
  if (!value) return undefined;
  const normalized = value.replace(/\s+/g, " ").trim();
  if (!normalized) return undefined;
  return normalized.slice(0, maxLength);
};

export const sanitizeKuerzel = (value: string): string | null => {
  const cleaned = value.trim().replace(/[^a-zA-Z0-9._-]/g, "-").slice(0, 30);
  return cleaned || null;
};
