import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SmartFlow — Corridor Operations",
  description:
    "Live control dashboard for the SmartFlow multi-agent traffic-signal digital twin.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
