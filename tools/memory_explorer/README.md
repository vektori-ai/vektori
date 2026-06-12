# Memory Explorer

Local, **read-only** UI for inspecting what your agent remembers — across both engines:

- **sentex** — the 3-layer conversation store (sentences → facts → episodes)
- **memfs** — the filesystem memory tree (markdown canonical)

Zero dependencies (Python stdlib + one HTML file, no CDN). Stores are opened
read-only at the SQLite level; nothing the UI does can mutate memory.

```bash
python -m tools.memory_explorer.server
# → http://127.0.0.1:8765
# defaults: --db ~/.vektori/vektori.db  --memfs-root ~/.vektori/memfs/default
```

What it answers:

| Question | Where |
|---|---|
| What does my agent know? | **Overview** — living profile + MEMORY.md side by side |
| What does the whole thing look like? | **Graph** — sentex as a layered hierarchy (sentences → facts → episodes, supersessions in red), memfs as a wikilink map clustered by directory |
| How did knowledge change over time? | **Facts → timeline** — one lane per subject, dashed arrows from superseded facts to their replacements |
| Why does it believe X? | Click any fact → verbatim source sentences, episodes, supersession chain |
| What did this conversation teach it? | **Conversations** — sentences that produced facts are marked; click for reverse provenance |
| What's in the file memory? | **Files** — tree, notes with clickable `[[wikilinks]]` + backlinks |
| What happened recently? | **Journal** — memfs op log, newest first |
| Anything about Y? | `/` — one search across both engines |

Engine identity is color-coded everywhere: **blue = sentex** (conversation memory),
**green = memfs** (file memory) — nav dots, section chips, search result groups.

Keyboard: `/` search · `1–7` switch sections · `Esc` close inspector.

Design rationale: see [DESIGN.md](DESIGN.md).
