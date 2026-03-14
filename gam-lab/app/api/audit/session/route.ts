import { cookies, headers } from "next/headers";
import { NextResponse } from "next/server";
import { appendAuditRecords, createServerUserId, sanitizeUserId } from "../_lib";
import { AUDIT_USER_COOKIE } from "../../../gam-lab/lib/audit";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => ({}))) as { preferredUserId?: string };
  const cookieStore = await cookies();
  const headerStore = await headers();
  const cookieUserId = sanitizeUserId(cookieStore.get(AUDIT_USER_COOKIE)?.value);
  const preferredUserId = sanitizeUserId(payload.preferredUserId);
  const userId = preferredUserId ?? cookieUserId ?? createServerUserId();

  await appendAuditRecords([
    {
      recordedAt: new Date().toISOString(),
      category: "system",
      action: "session.identified",
      userId,
      detail: {
        preferredUserId: preferredUserId ?? null,
        hadCookie: Boolean(cookieUserId),
        userAgent: headerStore.get("user-agent") ?? null,
      },
    },
  ]);

  const response = NextResponse.json({ userId });
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
