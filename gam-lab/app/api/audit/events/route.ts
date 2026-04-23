import { NextResponse } from "next/server";
import { appendAuditRecords, readAuditRecords, sanitizeUserId } from "../_lib";
import { type AuditQueuedEvent } from "../../../gam-lab/lib/audit";

export const runtime = "nodejs";

type RequestBody = {
  userId?: string;
  sessionId?: string;
  events?: AuditQueuedEvent[];
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const day = searchParams.get("day") ?? undefined;
  const userId = sanitizeUserId(searchParams.get("userId"));
  const sessionId = sanitizeUserId(searchParams.get("sessionId"));
  const category = searchParams.get("category")?.trim() || undefined;
  const action = searchParams.get("action")?.trim() || undefined;
  const query = searchParams.get("query")?.trim() || undefined;
  const limitParam = Number.parseInt(searchParams.get("limit") ?? "", 10);
  const limit = Number.isFinite(limitParam) ? limitParam : undefined;
  const result = await readAuditRecords({ day, userId, sessionId, category, action, query, limit });

  return NextResponse.json({
    ...result,
    summary: {
      categories: [...new Set(result.records.map((record) => record.category).filter(Boolean))],
      actions: [...new Set(result.records.map((record) => record.action).filter(Boolean))],
      users: [...new Set(result.records.map((record) => record.userId).filter(Boolean))],
      sessions: [...new Set(result.records.map((record) => record.sessionId).filter(Boolean))],
    },
  });
}

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => ({}))) as RequestBody;
  const userId = sanitizeUserId(payload.userId) ?? "unknown";
  const sessionId = sanitizeUserId(payload.sessionId) ?? `session-${crypto.randomUUID()}`;
  const events = Array.isArray(payload.events) ? payload.events.slice(0, 250) : [];

  if (!events.length) {
    return NextResponse.json({ stored: 0 }, { status: 400 });
  }

  const recordedAt = new Date().toISOString();
  const requestUrl = new URL(request.url);
  const records = events.map((event) => ({
    recordedAt,
    occurredAt: event.occurredAt,
    eventId: event.eventId,
    userId,
    sessionId,
    category: event.category,
    action: event.action,
    featureKey: event.featureKey ?? null,
    page: event.page ?? null,
    detail: event.detail ?? null,
    request: { route: requestUrl.pathname },
  }));

  await appendAuditRecords(records);

  return NextResponse.json({ stored: records.length, userId, sessionId });
}
