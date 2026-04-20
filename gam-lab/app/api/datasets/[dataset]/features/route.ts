const TRAINER_URL = process.env.TRAINER_URL ?? process.env.NEXT_PUBLIC_TRAINER_URL ?? "http://localhost:4001";

type Params = { dataset: string };

export async function GET(request: Request, context: { params: Promise<Params> }) {
  const { dataset } = await context.params;
  const url = new URL(request.url);
  const seed = url.searchParams.get("seed");
  const upstreamUrl = new URL(`${TRAINER_URL}/datasets/${encodeURIComponent(dataset)}/features`);
  if (seed) upstreamUrl.searchParams.set("seed", seed);

  const upstream = await fetch(upstreamUrl);
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
