const TRAINER_URL = process.env.TRAINER_URL ?? process.env.NEXT_PUBLIC_TRAINER_URL ?? "http://localhost:4001";

type Params = { name: string };

export async function GET(_request: Request, context: { params: Promise<Params> }) {
  const { name } = await context.params;
  const upstream = await fetch(`${TRAINER_URL}/models/${encodeURIComponent(name)}`);
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
