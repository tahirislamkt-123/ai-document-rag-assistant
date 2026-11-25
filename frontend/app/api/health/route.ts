import { proxyJson } from "@/lib/server-proxy";

export const dynamic = "force-dynamic";

export function GET() {
  return proxyJson("health");
}
