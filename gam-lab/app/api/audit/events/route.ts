import { cookies, headers } from "next/headers";
import { NextResponse } from "next/server";
import { appendAuditRecords, createServerUserId, sanitizeUserId } from "../_lib";
import { AUDIT_USER_COOKIE, type AuditQueuedEvent } from "../../../gam-lab/lib/audit";

export const runtime = "nodejs";

type RequestBody = {
  userId?: string;
  sessionId?: string;
  participantId?: string;
  events?: AuditQueuedEvent[];
};

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => ({}))) as RequestBody;
  const cookieStore = await cookies();
  const headerStore = await headers();
  const userId =
    sanitizeUserId(payload.userId) ??
    sanitizeUserId(cookieStore.get(AUDIT_USER_COOKIE)?.value) ??
    createServerUserId();
  const sessionId = sanitizeUserId(payload.sessionId) ?? `session-${crypto.randomUUID()}`;
  const participantId = sanitizeUserId(payload.participantId);
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
    participantId: participantId ?? null,
    sessionId,
    category: event.category,
    action: event.action,
    featureKey: event.featureKey ?? null,
    page: event.page ?? null,
    detail: event.detail ?? null,
    request: {
      route: requestUrl.pathname,
      referrer: headerStore.get("referer") ?? null,
      userAgent: headerStore.get("user-agent") ?? null,
    },
  }));

  await appendAuditRecords(records);

  const response = NextResponse.json({ stored: records.length, userId, sessionId });
  response.cookies.set({
    name: AUDIT_USER_COOKIE,
    value: userId,
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    maxAge: 60 * 60 * 24 * 365,
  });
  return response;
}
