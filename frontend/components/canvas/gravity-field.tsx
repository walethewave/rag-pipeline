"use client";

import { Bloom, EffectComposer, Vignette } from "@react-three/postprocessing";
import { Canvas, extend, useFrame, useThree, type ThreeElement } from "@react-three/fiber";
import { shaderMaterial } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useTheme } from "@/lib/theme";

/**
 * A near-Keplerian accretion disk: angular speed falls off with radius (inner orbit fast, outer
 * slow). The cursor exerts a gravity-like pull via an inverse-square warp in the vertex shader —
 * the field bends toward the pointer, reads as mass. Bloom fires ONLY on the inner core (which
 * the fragment shader artificially brightens past the luminanceThreshold); outer particles stay
 * as crisp, dim star-like dots. That separation is what makes it look like a glowing disk rather
 * than one overexposed cloud.
 */
const PARTICLE_COUNT = 6000;

const GravityMaterial = shaderMaterial(
  {
    uTime: 0,
    uMouse: new THREE.Vector2(20, 20), // far off-screen — no pull until the pointer moves
    uMouseStrength: 0.0,
    uColorInner: new THREE.Color("#d4ff5a"), // hot lime core
    uColorOuter: new THREE.Color("#3a5cff"), // cool indigo outer arm
  },
  /* -------------------------------------------------- vertex */ `
    uniform float uTime;
    uniform vec2  uMouse;
    uniform float uMouseStrength;

    attribute float aAngle0;
    attribute float aRadius;
    attribute float aSpeed;
    attribute float aSize;
    attribute float aSeed;

    varying float vRadiusNorm;
    varying float vAlpha;

    void main() {
      float angle = aAngle0 + uTime * aSpeed;

      // Elliptical disk: compress Y strongly so it reads as a clearly tilted plane
      vec3 pos = vec3(
        cos(angle) * aRadius,
        sin(angle) * aRadius * 0.26,
        sin(aSeed * 9.3 + uTime * 0.08) * 0.3
      );

      // Cursor gravity: inverse-square pull, soft min on distance to avoid singularity
      vec2 toMouse = uMouse - pos.xy;
      float dist    = length(toMouse) + 0.9;
      pos.xy       += normalize(toMouse) * (1.0 / (dist * dist)) * uMouseStrength * 9.0;

      vRadiusNorm = clamp(aRadius / 8.5, 0.0, 1.0);

      // Two-zone brightness: inner core is "lit", outer disk is dim.
      // This prevents bloom from eating the whole field.
      float innerBright = 1.0 - smoothstep(0.0, 0.28, vRadiusNorm);
      float outerFade   = 1.0 - smoothstep(0.42, 1.0,  vRadiusNorm);
      vAlpha = (0.012 + 0.09 * innerBright) * outerFade;

      vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
      // Smaller divisor = smaller screen-space points; avoids blob inflation in headless/HiDPI
      gl_PointSize    = aSize * (95.0 / -mvPosition.z);
      gl_Position     = projectionMatrix * mvPosition;
    }
  `,
  /* ------------------------------------------------ fragment */ `
    uniform vec3  uColorInner;
    uniform vec3  uColorOuter;
    varying float vRadiusNorm;
    varying float vAlpha;

    void main() {
      vec2  uv   = gl_PointCoord - 0.5;
      float d    = length(uv);
      // Tight radial falloff: each particle is a sharp star-point, not a blurry sphere
      float core = smoothstep(0.5, 0.04, d);
      if (core <= 0.002) discard;

      vec3 color = mix(uColorInner, uColorOuter, vRadiusNorm);
      // Boost inner brightness past bloom's luminanceThreshold (0.72).
      // Outer particles (vRadiusNorm → 1) multiply by ~0.75 — they stay below the threshold.
      color *= mix(2.9, 0.75, vRadiusNorm);

      gl_FragColor = vec4(color, core * vAlpha);
    }
  `
);

extend({ GravityMaterial });

declare module "@react-three/fiber" {
  interface ThreeElements {
    gravityMaterial: ThreeElement<typeof GravityMaterial>;
  }
}

function ParticleDisk({
  intensity = 1,
  colorInner,
  colorOuter,
}: {
  intensity?: number;
  colorInner: string;
  colorOuter: string;
}) {
  const materialRef = useRef<InstanceType<typeof GravityMaterial>>(null);
  const { viewport } = useThree();

  const geometry = useMemo(() => {
    const angle0    = new Float32Array(PARTICLE_COUNT);
    const radius    = new Float32Array(PARTICLE_COUNT);
    const speed     = new Float32Array(PARTICLE_COUNT);
    const size      = new Float32Array(PARTICLE_COUNT);
    const seed      = new Float32Array(PARTICLE_COUNT);
    const positions = new Float32Array(PARTICLE_COUNT * 3);

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // Power-law density: dense inner core, sparse outer tail — like a real accretion disk
      const r      = 0.45 + Math.pow(Math.random(), 0.58) * 8.2;
      angle0[i]    = Math.random() * Math.PI * 2;
      radius[i]    = r;
      // Keplerian angular speed; mix of CW/CCW adds visual depth without symmetry
      speed[i]     = (0.035 + 0.42 / (r + 0.4)) * (Math.random() > 0.5 ? 1 : -1) * 0.6;
      size[i]      = 0.55 + Math.random() * 1.6;  // dots, not blobs
      seed[i]      = Math.random() * 10.0;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("aAngle0",  new THREE.BufferAttribute(angle0,    1));
    geo.setAttribute("aRadius",  new THREE.BufferAttribute(radius,    1));
    geo.setAttribute("aSpeed",   new THREE.BufferAttribute(speed,     1));
    geo.setAttribute("aSize",    new THREE.BufferAttribute(size,      1));
    geo.setAttribute("aSeed",    new THREE.BufferAttribute(seed,      1));
    return geo;
  }, []);

  useFrame((state, delta) => {
    const mat = materialRef.current;
    if (!mat) return;
    mat.uTime += delta;

    const tx = (state.pointer.x * viewport.width)  / 2;
    const ty = (state.pointer.y * viewport.height) / 2;
    // Slow, heavy lerp — cursor influence feels gravitational, not snappy
    mat.uMouse.x       += (tx        - mat.uMouse.x)        * 0.045;
    mat.uMouse.y       += (ty        - mat.uMouse.y)        * 0.045;
    mat.uMouseStrength += (intensity - mat.uMouseStrength)   * 0.022;
  });

  return (
    <points geometry={geometry} rotation={[0.58, 0.12, 0.22]}>
      <gravityMaterial
        ref={materialRef}
        uColorInner={new THREE.Color(colorInner)}
        uColorOuter={new THREE.Color(colorOuter)}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

/**
 * Fixed full-viewport background. z-0, NOT z-[-10]: a negative z-index paints below the body's
 * own opaque background layer, hiding the canvas entirely even when WebGL renders correctly.
 * All foreground content (Nav, sections) must use z-10+ to stack above this.
 */
/** Theme-dependent look: dark mode keeps the original bright-on-black accretion disk; light mode
 * swaps in a dim bg and darker, less-bloomed particles so the field reads as a quiet watermark
 * instead of blowing out against a pale background. */
const THEME_LOOK = {
  dark: {
    bg: "#050608",
    colorInner: "#d4ff5a",
    colorOuter: "#3a5cff",
    bloomIntensity: 1.3,
    luminanceThreshold: 0.72,
    vignetteDarkness: 0.94,
  },
  light: {
    bg: "#f5f4ef",
    colorInner: "#5c7a1e",
    colorOuter: "#2a3a7a",
    bloomIntensity: 0.35,
    luminanceThreshold: 0.88,
    vignetteDarkness: 0.18,
  },
} as const;

export function GravityField({ intensity = 1 }: { intensity?: number }) {
  const { theme } = useTheme();
  const look = THEME_LOOK[theme];

  return (
    <div className="pointer-events-none fixed inset-0 z-0">
      <Canvas
        camera={{ position: [0, 0, 8.2], fov: 48 }}
        dpr={[1, 1.5]}
        gl={{ antialias: false, alpha: false }}
      >
        <color attach="background" args={[look.bg]} />
        <ParticleDisk intensity={intensity} colorInner={look.colorInner} colorOuter={look.colorOuter} />
        <EffectComposer multisampling={0}>
          {/*
            luminanceThreshold=0.72: the inner core writes color * 2.9 ≈ bright enough to cross
            this bar and get the glow halo. The dim outer particles write color * 0.75 and stay
            below — so they remain crisp dots, not contributing to the blurry-cloud look.
            Light mode raises the threshold and lowers intensity so the same particles don't
            blow out into a glare against a pale background.
          */}
          <Bloom
            intensity={look.bloomIntensity}
            luminanceThreshold={look.luminanceThreshold}
            luminanceSmoothing={0.12}
            mipmapBlur
          />
          <Vignette eskil={false} offset={0.1} darkness={look.vignetteDarkness} />
        </EffectComposer>
      </Canvas>
    </div>
  );
}
