"use client";

import { useCallback, useEffect, useState } from "react";

export type Theme = "dark" | "light";

const STORAGE_KEY = "vest-theme";

/** Reads/writes the theme to localStorage and to `<html data-theme="...">`, which globals.css
 * keys all its color variables off of. Call `setTheme`'s toggle to flip it; the DOM attribute
 * (and therefore every themed color on the page) updates immediately. */
export function useTheme() {
  const [theme, setThemeState] = useState<Theme>("dark");

  useEffect(() => {
    const stored = window.localStorage.getItem(STORAGE_KEY) as Theme | null;
    if (stored === "light" || stored === "dark") setThemeState(stored);
  }, []);

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    window.localStorage.setItem(STORAGE_KEY, next);
    document.documentElement.setAttribute("data-theme", next);
  }, []);

  const toggle = useCallback(() => {
    setTheme(theme === "dark" ? "light" : "dark");
  }, [theme, setTheme]);

  return { theme, setTheme, toggle };
}

/** Inline script string, inlined into <head> by layout.tsx (not a hook — runs before React
 * hydrates) so the correct theme applies before the first paint. Without this, the page would
 * flash dark-then-light (or vice versa) on every reload for anyone who picked a theme. */
export const THEME_INIT_SCRIPT = `
(function () {
  try {
    var t = localStorage.getItem("${STORAGE_KEY}");
    if (t === "light") document.documentElement.setAttribute("data-theme", "light");
  } catch (e) {}
})();
`;
