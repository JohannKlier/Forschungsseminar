const TRAINER_URL = process.env.TRAINER_URL ?? process.env.NEXT_PUBLIC_TRAINER_URL ?? "http://localhost:4001";

export async function POST(request: Request) {
  const body = await request.json();
  const upstream = await fetch(`${TRAINER_URL}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await readUpstreamBody(upstream);
  return new Response(payload.body, {
    status: upstream.status,
    headers: payload.headers,
  });
}

const readUpstreamBody = async (response: Response) => {
  const contentType = response.headers.get("content-type") ?? "application/json";
  const body = await response.text();
  return { body, headers: { "Content-Type": contentType } };
};
