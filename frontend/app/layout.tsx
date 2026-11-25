import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lattice — Document assistant",
  description:
    "A private, local-first assistant for asking questions across your documents.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
