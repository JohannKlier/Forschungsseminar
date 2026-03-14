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
  participantId?: string;
};

export type AuditLogFn = (event: AuditEventInput) => void;

export const AUDIT_USER_COOKIE = "gam-lab-user-id";
export const AUDIT_SESSION_STORAGE_KEY = "gam-lab-session-id";

export const createAuditId = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `audit-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

export const getParticipantIdFromLocation = () => {
  if (typeof window === "undefined") return undefined;
  const raw = new URLSearchParams(window.location.search).get("pid")?.trim();
  return raw ? raw : undefined;
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
