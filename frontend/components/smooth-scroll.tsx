"use client";

import Lenis from "lenis";
import { useEffect } from "react";

/**
 * Wraps the app in Lenis smooth-scroll - eases the scroll position toward its target instead of
 * jumping, which is most of what makes a dark/cinematic site feel "designed" rather than default.
 * Pure side-effect component: renders nothing, just drives the scroll loop for its lifetime.
 */
export function SmoothScroll() {
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.1,
      easing: (t: number) => Math.min(1, 1 - Math.pow(2, -10 * t)),
      smoothWheel: true,
    });

    let frameId: number;
    function raf(time: number) {
      lenis.raf(time);
      frameId = requestAnimationFrame(raf);
    }
    frameId = requestAnimationFrame(raf);

    return () => {
      cancelAnimationFrame(frameId);
      lenis.destroy();
    };
  }, []);

  return null;
}
