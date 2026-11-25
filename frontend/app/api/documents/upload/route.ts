import { proxyUpload } from "@/lib/server-proxy";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export function POST(request: Request) {
  return proxyUpload(request);
}
