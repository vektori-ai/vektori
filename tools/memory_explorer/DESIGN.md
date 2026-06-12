# Memory Explorer — design notes

Local, read-only inspector for both vektori memory engines. One question drives the
whole design: **"what does my agent know about me, and why does it think that?"**

## First principles

A person opening this tool has one of five jobs:

1. **Trust** — "what does it know?" → answerable in <5s without clicking anything.
2. **Provenance** — "why does it believe X?" → fact → episode → the verbatim sentence.
3. **Audit** — "what's wrong/stale?" → superseded chains, low-confidence, broken links.
4. **Find** — "what do you have about Y?" → one search across both engines.
5. **Change awareness** — "what did it learn recently?" → time-ordered writes.

Non-goals: editing (read-only by design — destructive ops belong to the CLI/API),
analytics dashboards, force-directed graph art. A graph hairball answers none of the
five jobs; *provenance chains on demand* do.

Two engines, one mental model: users think "memory", not storage backends. The
left rail is organized by **kind of memory** (facts, episodes, conversations, files,
journal), not by engine. Engine identity stays visible in paths/status only.

- **sentex** (3-layer): sentences → facts → episodes. Temporal, per-user, lineage-rich.
- **memfs**: markdown tree + wikilinks. Spatial, file-shaped, grep-able.

The two meet in: Overview (living profile ⊕ MEMORY.md) and Search (both engines).

## HCI principles applied

- **Overview first, zoom + filter, details on demand** (Shneiderman). Overview page is
  the trust answer; lists filter; the inspector is the only place detail appears.
- **Master–detail, one focus.** Three panes: rail / list / inspector. Never six
  competing cards. Esc closes the inspector; selection is always visible.
- **Recognition over recall.** Counts on every rail item; filters are visible chips,
  not hidden menus; provenance is click-through, not query syntax.
- **Direct manipulation + reversibility.** Click a fact → its sources; click a
  sentence → the facts it produced (reverse provenance — the debugging direction).
  Click a wikilink → that note. Back/forward = browser history.
- **Status visibility.** DB path, memfs root, index freshness, last-sync, counts —
  in the top bar, always.
- **Keyboard.** `/` search, `1–6` sections, `j/k` move, `Enter` inspect, `Esc` close.
- **Honest empty states.** Each view explains how data gets there when it's empty.

## Visual language (anti-slop rules)

- Graphite neutrals (`#101113` bg, 1px `#26282c` borders), one accent (`#6aa1ff`),
  used only for selection/links/focus. No gradients, no glow shadows, no purple.
- `system-ui` text, `ui-monospace` for IDs, paths, quotes. 13px base, 36px rows.
- Color = meaning only: layer badges (fact/episode/sentence), note-type chips,
  confidence as a 36px inline meter, state by dimming (superseded), not by rainbow.
- Density over chrome: this is a developer instrument, closer to a debugger than a
  marketing dashboard.

## Architecture

- `server.py` — stdlib `ThreadingHTTPServer`, opens both SQLite stores in
  `mode=ro`, reads memfs markdown directly (files are canonical; explorer must work
  even if `.memfs/index.db` was deleted). Zero dependencies.
- `memory_explorer.html` — single file, vanilla JS, zero CDN/network. Hash-based
  routing so provenance links are real URLs.
- Read-only is enforced at the connection level, not by convention.
