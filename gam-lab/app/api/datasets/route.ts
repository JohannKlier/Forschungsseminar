const TRAINER_URL = process.env.TRAINER_URL ?? process.env.NEXT_PUBLIC_TRAINER_URL ?? "http://localhost:4001";

export async function GET() {
  const upstream = await fetch(`${TRAINER_URL}/datasets`, { cache: "no-store" });
  const contentType = upstream.headers.get("content-type") ?? "application/json";
  const body = await upstream.text();
  return new Response(body, { status: upstream.status, headers: { "Content-Type": contentType } });
}
