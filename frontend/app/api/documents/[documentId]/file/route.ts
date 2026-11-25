import { proxyPdf } from "@/lib/server-proxy";

export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  context: { params: Promise<{ documentId: string }> },
) {
  const { documentId } = await context.params;
  return proxyPdf(documentId);
}
