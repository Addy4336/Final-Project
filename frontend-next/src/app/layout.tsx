import type { Metadata } from "next";
import { DM_Serif_Display, IBM_Plex_Mono, DM_Sans } from "next/font/google";
import "./globals.css";

const dmSerifDisplay = DM_Serif_Display({
  weight: ["400"],
  style: ["normal", "italic"],
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

const ibmPlexMono = IBM_Plex_Mono({
  weight: ["400", "500", "600"],
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

const dmSans = DM_Sans({
  weight: ["300", "400", "500"],
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

export const metadata: Metadata = {
  title: "VisionQuery — Multimodal Research Console",
  description:
    "PhD-grade multimodal AI research instrument combining OCR, VQA, and Satellite analysis with hybrid reasoning.",
  keywords: ["VQA", "OCR", "satellite analysis", "multimodal AI", "research"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`dark ${dmSerifDisplay.variable} ${ibmPlexMono.variable} ${dmSans.variable}`}>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
