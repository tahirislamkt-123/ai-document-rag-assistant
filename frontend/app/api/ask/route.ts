import { proxyAsk } from "@/lib/server-proxy";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  return proxyAsk(request);
}
