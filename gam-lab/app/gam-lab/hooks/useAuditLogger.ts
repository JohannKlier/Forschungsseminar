"use client";

import { useCallback, useEffect, useRef } from "react";
import {
  AUDIT_SESSION_STORAGE_KEY,
  createAuditId,
  getCurrentPage,
  getParticipantIdFromLocation,
  type AuditIdentity,
  type AuditEventInput,
  type AuditLogFn,
  type AuditQueuedEvent,
} from "../lib/audit";

const FLUSH_DELAY_MS = 800;
const MAX_BATCH_SIZE = 50;

const getOrCreateSessionId = () => {
  if (typeof window === "undefined") {
    return `session-${createAuditId()}`;
  }
  const existing = window.sessionStorage.getItem(AUDIT_SESSION_STORAGE_KEY);
  if (existing) return existing;
  const created = `session-${createAuditId()}`;
  window.sessionStorage.setItem(AUDIT_SESSION_STORAGE_KEY, created);
  return created;
};

export function useAuditLogger() {
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
          participantId: identity.participantId,
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
        flushTimerRef.current = window.setTimeout(() => {
          void flush();
        }, FLUSH_DELAY_MS);
      }
    }
  }, []);

  const scheduleFlush = useCallback(() => {
    if (typeof window === "undefined") return;
    if (flushTimerRef.current != null) {
      window.clearTimeout(flushTimerRef.current);
    }
    flushTimerRef.current = window.setTimeout(() => {
      void flush();
    }, FLUSH_DELAY_MS);
  }, [flush]);

  const logEvent = useCallback<AuditLogFn>(
    (event: AuditEventInput) => {
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

  useEffect(() => {
    let cancelled = false;
    const participantId = getParticipantIdFromLocation();
    const sessionId = getOrCreateSessionId();

    const init = async () => {
      try {
        const response = await fetch("/api/audit/session", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ preferredUserId: participantId }),
        });
        const payload = (await response.json()) as { userId?: string };
        if (cancelled) return;
        identityRef.current = {
          userId: payload.userId ?? `anon-${createAuditId()}`,
          sessionId,
          participantId,
        };
      } catch {
        if (cancelled) return;
        identityRef.current = {
          userId: participantId ?? `anon-${createAuditId()}`,
          sessionId,
          participantId,
        };
      }

      logEvent({
        category: "system",
        action: "session.started",
        detail: {
          participantId: participantId ?? null,
          pathname: window.location.pathname,
        },
      });
      void flush();
    };

    void init();

    const handlePageHide = () => {
      const identity = identityRef.current;
      if (!identity || queueRef.current.length === 0 || typeof navigator.sendBeacon !== "function") {
        return;
      }
      const pending = queueRef.current.splice(0, queueRef.current.length);
      navigator.sendBeacon(
        "/api/audit/events",
        new Blob(
          [
            JSON.stringify({
              userId: identity.userId,
              sessionId: identity.sessionId,
              participantId: identity.participantId,
              events: pending,
            }),
          ],
          { type: "application/json" },
        ),
      );
    };

    window.addEventListener("pagehide", handlePageHide);
    return () => {
      cancelled = true;
      if (flushTimerRef.current != null) {
        window.clearTimeout(flushTimerRef.current);
      }
      window.removeEventListener("pagehide", handlePageHide);
    };
  }, [flush, logEvent]);

  return { logEvent };
}
