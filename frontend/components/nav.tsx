"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import { useTheme } from "@/lib/theme";

const LINKS = [
  { href: "/", label: "Ingest" },
  { href: "/chat", label: "Ask" },
];

export function Nav() {
  const pathname = usePathname();
  const { theme, toggle } = useTheme();

  return (
    <header className="fixed inset-x-0 top-0 z-50 flex items-center justify-between px-6 py-5 md:px-10">
      <Link href="/" className="font-mono text-[13px] tracking-[0.25em] text-ink uppercase">
        Vest<span className="text-accent">.</span>
      </Link>
      <div className="flex items-center gap-2">
        <nav className="flex items-center gap-1 rounded-full border border-line/80 bg-bg-elevated/60 px-1.5 py-1.5 backdrop-blur-md">
          {LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={clsx(
                  "rounded-full px-4 py-1.5 font-mono text-[11px] uppercase tracking-[0.18em] transition-colors",
                  active ? "bg-ink text-bg" : "text-ink-dim hover:text-ink"
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
        <button
          onClick={toggle}
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          className="flex h-9 w-9 items-center justify-center rounded-full border border-line/80 bg-bg-elevated/60 text-ink-dim backdrop-blur-md transition-colors hover:text-ink"
        >
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>
    </header>
  );
}

function SunIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="4.5" stroke="currentColor" strokeWidth="1.6" />
      <path
        d="M12 2.5v3M12 18.5v3M3.5 12h3M17.5 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"
        stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"
      />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
      <path
        d="M20.5 14.5A8.5 8.5 0 119.5 3.5a7 7 0 1011 11z"
        stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round"
      />
    </svg>
  );
}
