import { useCallback, useEffect, useRef } from "react";

export type LogEntry = {
  ts: number;
  elapsed: number; // ms since session start
  action: string;
  feature?: string;
  detail?: Record<string, unknown>;
};

type StudyLog = {
  participantId: string;
  sessionStart: number;
  entries: LogEntry[];
};

const STORAGE_KEY = "gam-lab-study-log";

/**
 * Comprehensive study logger that records every user action.
 * Persists to localStorage and supports JSON download.
 */
export function useStudyLogger() {
  const participantId =
    typeof window !== "undefined"
      ? new URLSearchParams(window.location.search).get("pid") ?? "unknown"
      : "unknown";

  const logRef = useRef<StudyLog>({
    participantId,
    sessionStart: Date.now(),
    entries: [],
  });

  // Restore any prior log for the same participant on mount.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const prev: StudyLog = JSON.parse(raw);
        if (prev.participantId === participantId) {
          logRef.current = prev;
          return;
        }
      }
    } catch {
      /* ignore parse errors */
    }
    logRef.current = {
      participantId,
      sessionStart: Date.now(),
      entries: [],
    };
  }, [participantId]);

  // Persist to localStorage (debounced via flush-on-write).
  const persist = useCallback(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(logRef.current));
    } catch {
      /* quota exceeded — best effort */
    }
  }, []);

  /** Core logging function. Call this for every trackable event. */
  const log = useCallback(
    (action: string, feature?: string, detail?: Record<string, unknown>) => {
      const now = Date.now();
      const entry: LogEntry = {
        ts: now,
        elapsed: now - logRef.current.sessionStart,
        action,
        ...(feature !== undefined && { feature }),
        ...(detail !== undefined && { detail }),
      };
      logRef.current.entries.push(entry);
      persist();
    },
    [persist],
  );

  /** Download the full log as a JSON file. */
  const downloadLog = useCallback(() => {
    const blob = new Blob([JSON.stringify(logRef.current, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `study-log-${participantId}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [participantId]);

  /** Get the current log (read-only snapshot). */
  const getLog = useCallback((): StudyLog => {
    return logRef.current;
  }, []);

  /** Clear the log (e.g. for a fresh session). */
  const clearLog = useCallback(() => {
    logRef.current = {
      participantId,
      sessionStart: Date.now(),
      entries: [],
    };
    persist();
  }, [participantId, persist]);

  return { participantId, log, downloadLog, getLog, clearLog };
}
