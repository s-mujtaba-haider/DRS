import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";

import BackendStatus from "@/components/BackendStatus";

import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "DRS Review Console",
  description: "Experimental cricket ball-tracking and decision-review research prototype",
};

const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/matches", label: "Match day" },
  { href: "/capture", label: "Live capture" },
  { href: "/upload", label: "New analysis" },
  { href: "/sessions", label: "Sessions" },
  { href: "/datasets", label: "Datasets" },
  { href: "/system", label: "System" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${geistSans.variable} ${geistMono.variable} min-h-screen antialiased`}>
        <div className="flex min-h-screen flex-col md:flex-row">
          <aside className="border-b border-line bg-panel md:w-56 md:shrink-0 md:border-b-0 md:border-r">
            <div className="px-4 py-4">
              <div className="text-sm font-semibold tracking-wide">DRS Review Console</div>
              <div className="mt-0.5 text-[11px] uppercase tracking-wider text-warn">Research prototype</div>
            </div>
            <nav className="flex gap-1 overflow-x-auto px-2 pb-3 md:flex-col md:pb-0">
              {NAV.map((n) => (
                <Link key={n.href} href={n.href} className="whitespace-nowrap rounded px-3 py-1.5 text-sm text-muted hover:bg-panel-2 hover:text-ink">
                  {n.label}
                </Link>
              ))}
            </nav>
            <div className="hidden px-4 py-4 md:block">
              <BackendStatus />
            </div>
          </aside>
          <main className="min-w-0 flex-1 px-4 py-5 md:px-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
