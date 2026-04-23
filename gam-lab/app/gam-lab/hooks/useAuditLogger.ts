"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AUDIT_CODE_STORAGE_KEY,
  AUDIT_SESSION_STORAGE_KEY,
  createAuditId,
  getCurrentPage,
  sanitizeKuerzel,
  type AuditIdentity,
  type AuditEventInput,
  type AuditLogFn,
  type AuditQueuedEvent,
} from "../lib/audit";

const FLUSH_DELAY_MS = 800;
const MAX_BATCH_SIZE = 50;

const getOrCreateSessionId = () => {
  if (typeof window === "undefined") return `session-${createAuditId()}`;
  const existing = window.sessionStorage.getItem(AUDIT_SESSION_STORAGE_KEY);
  if (existing) return existing;
  const created = `session-${createAuditId()}`;
  window.sessionStorage.setItem(AUDIT_SESSION_STORAGE_KEY, created);
  return created;
};

const readStoredKuerzel = (): string | null => {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(AUDIT_CODE_STORAGE_KEY);
  return raw ? sanitizeKuerzel(raw) : null;
};

export function useAuditLogger() {
  const [kuerzel, setKuerzelState] = useState<string | null>(() => readStoredKuerzel());
  const queueRef = useRef<AuditQueuedEvent[]>([]);
  const identityRef = useRef<AuditIdentity | null>(null);
  const flushTimerRef = useRef<number | null>(null);
  const isFlushingRef = useRef(false);

  const flush = useCallback(async () => {
    if (isFlushingRef.current) return;
    const identity = identityRef.current;
    if (!identity || queueRef.current.length === 0) return;
    isFlushingRef.current = true;
    const batch = queueRef.current.splice(0, MAX_BATCH_SIZE);
    try {
      const response = await fetch("/api/audit/events", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: identity.userId,
          sessionId: identity.sessionId,
          events: batch,
        }),
        keepalive: true,
      });
      if (!response.ok) {
        queueRef.current = [...batch, ...queueRef.current];
      }
    } catch {
      queueRef.current = [...batch, ...queueRef.current];
    } finally {
      isFlushingRef.current = false;
      if (queueRef.current.length > 0) {
        flushTimerRef.current = window.setTimeout(() => void flush(), FLUSH_DELAY_MS);
      }
    }
  }, []);

  const scheduleFlush = useCallback(() => {
    if (typeof window === "undefined") return;
    if (flushTimerRef.current != null) window.clearTimeout(flushTimerRef.current);
    flushTimerRef.current = window.setTimeout(() => void flush(), FLUSH_DELAY_MS);
  }, [flush]);

  const logEvent = useCallback<AuditLogFn>(
    (event: AuditEventInput) => {
      if (!identityRef.current) return;
      queueRef.current.push({
        eventId: createAuditId(),
        occurredAt: new Date().toISOString(),
        page: event.page ?? getCurrentPage(),
        ...event,
      });
      scheduleFlush();
    },
    [scheduleFlush],
  );

  const setKuerzel = useCallback(
    (code: string) => {
      const sanitized = sanitizeKuerzel(code);
      if (!sanitized) return;
      window.localStorage.setItem(AUDIT_CODE_STORAGE_KEY, sanitized);
      const sessionId = getOrCreateSessionId();
      identityRef.current = { userId: sanitized, sessionId };
      setKuerzelState(sanitized);
      queueRef.current.push({
        eventId: createAuditId(),
        occurredAt: new Date().toISOString(),
        category: "system",
        action: "session.started",
        detail: { kuerzel: sanitized, pathname: window.location.pathname },
      });
      void flush();
    },
    [flush],
  );

  // On mount: restore stored kuerzel and start session
  useEffect(() => {
    const stored = readStoredKuerzel();
    if (!stored) return;
    const sessionId = getOrCreateSessionId();
    identityRef.current = { userId: stored, sessionId };
    // log without going through logEvent to avoid the identityRef timing issue
    queueRef.current.push({
      eventId: createAuditId(),
      occurredAt: new Date().toISOString(),
      category: "system",
      action: "session.started",
      detail: { kuerzel: stored, pathname: window.location.pathname },
    });
    void flush();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const handlePageHide = () => {
      const identity = identityRef.current;
      if (!identity || queueRef.current.length === 0 || typeof navigator.sendBeacon !== "function") return;
      const pending = queueRef.current.splice(0, queueRef.current.length);
      navigator.sendBeacon(
        "/api/audit/events",
        new Blob(
          [JSON.stringify({ userId: identity.userId, sessionId: identity.sessionId, events: pending })],
          { type: "application/json" },
        ),
      );
    };

    window.addEventListener("pagehide", handlePageHide);
    return () => {
      window.removeEventListener("pagehide", handlePageHide);
      if (flushTimerRef.current != null) window.clearTimeout(flushTimerRef.current);
    };
  }, []);

  return { logEvent, kuerzel, setKuerzel };
}
