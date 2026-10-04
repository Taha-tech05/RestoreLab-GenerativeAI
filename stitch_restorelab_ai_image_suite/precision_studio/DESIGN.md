---
name: Precision Studio
colors:
  surface: '#faf8ff'
  surface-dim: '#d2d9f4'
  surface-bright: '#faf8ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f3ff'
  surface-container: '#eaedff'
  surface-container-high: '#e2e7ff'
  surface-container-highest: '#dae2fd'
  on-surface: '#131b2e'
  on-surface-variant: '#464555'
  inverse-surface: '#283044'
  inverse-on-surface: '#eef0ff'
  outline: '#777587'
  outline-variant: '#c7c4d8'
  surface-tint: '#4d44e3'
  primary: '#3525cd'
  on-primary: '#ffffff'
  primary-container: '#4f46e5'
  on-primary-container: '#dad7ff'
  inverse-primary: '#c3c0ff'
  secondary: '#4648d4'
  on-secondary: '#ffffff'
  secondary-container: '#6063ee'
  on-secondary-container: '#fffbff'
  tertiary: '#005338'
  on-tertiary: '#ffffff'
  tertiary-container: '#006e4b'
  on-tertiary-container: '#67f4b7'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e2dfff'
  primary-fixed-dim: '#c3c0ff'
  on-primary-fixed: '#0f0069'
  on-primary-fixed-variant: '#3323cc'
  secondary-fixed: '#e1e0ff'
  secondary-fixed-dim: '#c0c1ff'
  on-secondary-fixed: '#07006c'
  on-secondary-fixed-variant: '#2f2ebe'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#faf8ff'
  on-background: '#131b2e'
  surface-variant: '#dae2fd'
typography:
  display-lg:
    fontFamily: Geist
    fontSize: 48px
    fontWeight: '600'
    lineHeight: 56px
    letterSpacing: -0.03em
  display-lg-mobile:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '500'
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: Geist
    fontSize: 18px
    fontWeight: '500'
    lineHeight: 26px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: -0.005em
  body-md:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  body-sm:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0em
  label-lg:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
    letterSpacing: 0.01em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '400'
    lineHeight: 14px
    letterSpacing: 0.04em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1.5rem
  gutter-mobile: 0.75rem
  margin: 2rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.5rem
---

## Brand & Style

This design system targets digital creators, archivists, machine learning practitioners, and creative technologists operating high-fidelity image restoration and generation models. 

The aesthetic is characterized as **Modern Technical Precision**—a synthesis of high-end developer tools and editorial creative suites. It combines surgical clarity with high aesthetic craft. The interface avoids frivolous ornamentation in favor of crisp delineation, structured metrics, and airy visual breathing room that lets visual outputs dominate the viewport.

Key qualities:
- **Atmosphere:** Controlled, luminous, analytical, and highly responsive.
- **Visual Rhythm:** Balanced light surfaces set against slate structures, punctuated by electric indigo/violet focal points that signal activity, compute, and precision state changes.
- **Texture:** Matte, anti-glare canvas tones overlaid with translucent panels, fine 1px hairline structural lines, and crisp monospace analytical telemetry.

## Colors

The palette employs a multi-tiered neutral canvas built from cool slate tones, keeping visual noise low while maintaining AA/AAA contrast ratios for dense computational data.

- **Primary (`#4F46E5` - Deep Indigo):** Used for primary conversion triggers, focal call-to-actions, active slider tracks, and prominent selected view states.
- **Secondary (`#6366F1` - Electric Violet-Indigo):** Drives interactive states, processing pulses, focus rings, hover indicators, and secondary computational highlights.
- **Tertiary (`#10B981` - Emerald Precision):** Dedicated entirely to runtime telemetry, success states, GPU compute availability, and live status badges.
- **Neutral (`#0F172A` - Slate Anchor):** Informs all high-priority typographic levels, deep borders, and structural framing.

### Surface Tiers
- **Base Canvas (`#F8FAFC`):** The master workbench background.
- **Surface Elevated (`#FFFFFF`):** Work panels, inspector decks, and comparison frames.
- **Surface Muted (`#F1F5F9`):** Input wells, inactive tracks, and chip backgrounds.
- **Micro-Border Subtle (`#E2E8F0`):** Default 1px bounding stroke for cards and containers.
- **Micro-Border Strong (`#CBD5E1`):** Active structural dividers, divider lines, and hover borders.

## Typography

Typography establishes an immediate dual-voice hierarchy: **Geist** conveys interface flow, tool options, and human-readable context with tight tracking and neutral geometric clarity; **JetBrains Mono** introduces deterministic instrumentation for numerical readouts, latency values, generation seeds, scale factors, and operational statuses.

- Headings and body copy rely on tightened negative letter-spacing to reinforce modern editorial discipline.
- JetBrains Mono is restricted strictly to machine metadata, processing prompts, resolution parameters (`4096x2160`), model tags, and telemetry badges.

## Layout & Spacing

The canvas is anchored by a structured 12-column fluid grid system on desktop, collapsing to 6 columns on tablet and a single fluid column on mobile. 

- **Layout Grid:** Desktop views use 24px (`1.5rem`) gutters with outer margins expanding up to 32px (`2rem`), accommodating a three-pane workbench layout (Parameters, Canvas/Viewport, History/Telemetry).
- **Responsive Handling:** On tablet, lateral tool docks turn into slide-out bottom sheets or drawer panels. Viewports maintain fixed aspect-ratio containers with dynamic fit algorithms.
- **Rhythm:** Spacing follows a 4px base increment. High-density controls sit tightly packed inside parent cards using `space-xs` and `space-sm`, while top-level studio partitions rely on `space-lg` and `space-xl` for strong structural separation.

## Elevation & Depth

This system intentionally relies on layered white/slate surfaces combined with hairline micro-borders rather than deep physical drop shadows.

- **Level 0 (Base Canvas):** `#F8FAFC`, flat.
- **Level 1 (Panels & Master Cards):** `#FFFFFF` with a 1px solid `#E2E8F0` micro-border and an ultra-diffused ambient shadow: `0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.02)`.
- **Level 2 (Popovers, Flyouts, Context Bars):** `#FFFFFF` with 1px solid `#CBD5E1` and ambient elevation: `0 10px 25px -5px rgba(15, 23, 42, 0.06), 0 8px 10px -6px rgba(15, 23, 42, 0.04)`.
- **Level 3 (Modal Dialogs & Zoom Overlays):** `#FFFFFF` with `0 25px 50px -12px rgba(15, 23, 42, 0.12)`.
- **Backdrop Blur:** Floating controls over viewports utilize `backdrop-filter: blur(12px)` paired with `rgba(255, 255, 255, 0.82)` background fills and `#E2E8F0` borders.

## Shapes

The design uses an intentional scale of soft corners (`roundedness: 2` base) scaled up to `rounded-xl` (16px) and `rounded-2xl` (24px) for primary containers and canvas frames.

- **Cards and Viewports:** Apply `rounded-2xl` (24px) to primary image previewers and master workbench containers; use `rounded-xl` (16px) for parameter groups and control sections.
- **Interactive Controls:** Buttons, inputs, and chips use `rounded-lg` (8px to 10px) to balance friendly approachability with structural discipline.
- **Status Indicators & Swatches:** Circular (`rounded-full`) for live indicator lights, split-view drag knobs, and model state chips.

## Components

### Buttons
- **Primary:** High-impact `#4F46E5` fill, `#FFFFFF` text, subtle inset highlight `box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2)`, `rounded-lg`. Hover transitions to `#4338CA`.
- **Secondary / Outline:** `#FFFFFF` background, 1px solid `#E2E8F0`, `#0F172A` text. On hover: border transitions to `#CBD5E1` with `#F8FAFC` fill.
- **Ghost / Tool Action:** Transparent background, slate icon/text. On hover: `#F1F5F9` background, `rounded-md`.

### Sliders (Upscaling, Denoise, Prompt Strength)
- **Track:** 4px thickness, `#E2E8F0` neutral fill with active fill in `#4F46E5`.
- **Thumb:** 18px circle, `#FFFFFF` fill with 1px border in `#CBD5E1` and shadow `0 2px 4px rgba(15, 23, 42, 0.1)`. Active/focused states project an electric ring: `0 0 0 4px rgba(99, 102, 241, 0.18)`.
- **Value Tag:** Displayed in `JetBrains Mono` (`label-sm`), anchored above or to the right of the track.

### Before/After Comparison Viewer
- Central dividing line: 2px wide `#FFFFFF` line backed by a soft edge shadow.
- Central grab handle: 32px floating pill or disc with `#FFFFFF` fill, 1px micro-border `#E2E8F0`, containing two subtle directional carats (`◂ ▸`).
- Corner watermark tags (`Original` / `Restored`) rendered as frosted badges (`backdrop-blur-md`, `bg-white/70`, `JetBrains Mono`, uppercase).

### Status Badges & Pulsing Nodes
- Pill-shaped badges (`rounded-full`) with a `#F1F5F9` background and `#E2E8F0` micro-border.
- Real-time online state: A 6px emerald circle (`#10B981`) next to a CSS keyframe pulse ring (`animate-ping`) with `rgba(16, 185, 129, 0.4)` opacity.
- Accompanying engine stats rendered in `label-sm` (`"A100 • 14ms Latency"`).

### Inputs & Selectors
- Default: `#FFFFFF` fill, 1px `#E2E8F0` micro-border, `rounded-lg`, `#0F172A` text, Geist Sans.
- Focused: Border shifts to `#6366F1` with an outer glow `0 0 0 3px rgba(99, 102, 241, 0.15)`.
- Metadata prompts and parameter controls include monospaced prefixes or suffixes (e.g., `cfg_scale:`, `seed: 489102`).

### Studio Cards
- Clean `#FFFFFF` fill, enclosed in 1px `#E2E8F0`, shaped with `rounded-xl` or `rounded-2xl`. Header blocks feature separated baseline hairpins dividing metadata metrics from the canvas.