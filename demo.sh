#!/usr/bin/env bash
# vektori demo script — for screen recording
# run: bash demo.sh
# tip: use a terminal with big font, dark theme, ~100 cols wide

set -euo pipefail

BOLD="\033[1m"
DIM="\033[2m"
GREEN="\033[32m"
CYAN="\033[36m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

# ── helpers ──────────────────────────────────────────────────────────────────

hr() { printf "${DIM}%0.s─${RESET}" $(seq 1 60); echo; }

type_cmd() {
    # prints a "typed" prompt then runs the command
    local cmd="$1"
    printf "\n${GREEN}❯${RESET} "
    for ((i=0; i<${#cmd}; i++)); do
        printf "%s" "${cmd:$i:1}"
        sleep 0.03
    done
    echo
    sleep 0.6
    eval "$cmd"
}

pause() { sleep "${1:-2}"; }

section() {
    echo
    hr
    printf "${BOLD}${CYAN}  $1${RESET}\n"
    hr
    pause 1
}

# ── env ──────────────────────────────────────────────────────────────────────

export VEKTORI_USER_ID="${VEKTORI_USER_ID:-dev}"
export VEKTORI_EXTRACTION_MODEL="${VEKTORI_EXTRACTION_MODEL:-litellm:groq/llama-3.3-70b-versatile}"
export VEKTORI_EMBEDDING_MODEL="${VEKTORI_EMBEDDING_MODEL:-sentence-transformers:all-MiniLM-L6-v2}"

# ── SCENE 1: the hook ─────────────────────────────────────────────────────────

clear
pause 1

echo
printf "${BOLD}  your AI assistant forgets everything after every session.${RESET}\n"
pause 2
printf "${DIM}  every project. every decision. every bug. gone.${RESET}\n"
pause 2
printf "${DIM}  mem0 charges you \$50/month to fix this.${RESET}\n"
pause 2
printf "${YELLOW}${BOLD}  we just read ~/.claude${RESET}\n"
pause 3

# ── SCENE 2: detect ──────────────────────────────────────────────────────────

section "STEP 1 — vektori detects your sessions"

type_cmd "vektori inject -u dev --list"
pause 3

# ── SCENE 3: inject ──────────────────────────────────────────────────────────

section "STEP 2 — inject everything. one command."

type_cmd "vektori inject -u dev --since 30 --yes"
pause 2

type_cmd "vektori stats -u dev"
pause 3

# ── SCENE 4: recall across sessions ──────────────────────────────────────────

section "STEP 3 — now ask anything. across every session."

echo
printf "${DIM}  (claude code and codex sessions. all projects. past 30 days.)${RESET}\n"
pause 2

type_cmd "vektori recall \"what was the engram project about?\" -u dev"
pause 3

type_cmd "vektori recall \"what did we build at the hackathon?\" -u dev"
pause 3

type_cmd "vektori recall \"what storage backends did we add to vektori?\" -u dev"
pause 3

# ── SCENE 5: L2 — the full story ─────────────────────────────────────────────

section "STEP 4 — L2: reconstruct the full conversation"

echo
printf "${DIM}  L0 = just facts.  L1 = facts + source sentences.  L2 = full story.${RESET}\n"
pause 2

type_cmd "vektori search \"vektori storage decisions\" -u dev --depth l2"
pause 4

# ── SCENE 6: cross-project ────────────────────────────────────────────────────

section "STEP 5 — cross-project memory"

type_cmd "vektori search \"what projects have we shipped\" -u dev --depth l1 --top-k 6"
pause 3

type_cmd "vektori search \"what bugs did codex fix on vektori\" -u dev --expand"
pause 3

# ── SCENE 7: the closer ───────────────────────────────────────────────────────

echo
hr
echo
printf "${BOLD}  every claude code session. every codex session.${RESET}\n"
pause 1
printf "${BOLD}  across every project. fully searchable.${RESET}\n"
pause 1
printf "${BOLD}  L0 / L1 / L2 depth. semantic search. zero cloud.${RESET}\n"
pause 2
echo
printf "${GREEN}${BOLD}  pip install vektori${RESET}\n"
echo
hr
echo
