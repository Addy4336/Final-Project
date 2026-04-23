import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VisionQuery Nexus — PhD Research Console",
  description:
    "Multimodal AI research console combining OCR, VQA, and Satellite analysis with hybrid reasoning. Built for PhD-level academic research.",
  keywords: ["VQA", "OCR", "satellite analysis", "multimodal AI", "research"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="surface-noise min-h-screen">{children}</body>
    </html>
  );
}
