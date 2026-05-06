#!/usr/bin/env bash
# Research team: 1:codex:scientist (gpt-5.5 xhigh) + 1:claude:analyst + 1:claude:critic (opus-4-7)
# Usage: ./scripts/research-team.sh "topic or research question"
# After completion, run /graphify in Claude Code with the output file path.

set -euo pipefail

TASK="${1:-}"
if [[ -z "$TASK" ]]; then
  echo "Usage: $0 \"<research topic or question>\"" >&2
  echo "" >&2
  echo "Roles:" >&2
  echo "  scientist (codex gpt-5.5 xhigh) — gather evidence, synthesize findings" >&2
  echo "  analyst   (claude-opus-4-7)      — structured deep analysis" >&2
  echo "  critic    (claude-opus-4-7)      — challenge assumptions, find weaknesses" >&2
  exit 1
fi

# Model env vars (omc.jsonc handles project config; these ensure workers inherit correct models)
export CLAUDE_MODEL="claude-opus-4-7"
export OMC_EXTERNAL_MODELS_DEFAULT_CODEX_MODEL="gpt-5.5"
# Note: model_reasoning_effort=xhigh already set globally in ~/.codex/config.toml

OUTDIR=".omc/research"
mkdir -p "$OUTDIR"

# Derive slug for monitoring and output file
SLUG=$(echo "$TASK" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | tr -s '-' | cut -c1-50 | sed 's/-$//')
OUTFILE="$OUTDIR/${SLUG}-knowledge.md"

echo "[research-team] Launching..."
echo "  scientist : codex gpt-5.5 + xhigh"
echo "  analyst   : claude-opus-4-7"
echo "  critic    : claude-opus-4-7"
echo "  task      : $TASK"
echo "  output    : $OUTFILE"
echo ""

omc team 1:codex:scientist,1:claude:analyst,1:claude:critic "$TASK"

echo ""
echo "[research-team] Team slug: $SLUG"
echo ""
echo "Monitor:  omc team status $SLUG"
echo ""
echo "When done, collect and graphify:"
echo "  omc team api get-summary --input '{\"team_name\":\"$SLUG\"}' --json \\"
echo "    | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d.get('summary','') or json.dumps(d,indent=2))\" \\"
echo "    > $OUTFILE"
echo "  # Then in Claude Code: /graphify $OUTFILE"
