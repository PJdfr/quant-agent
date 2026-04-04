#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILLS_DIR="$HOME/.claude/skills"

echo "==> Installing quant-agent dependencies..."
pip install -r "$REPO_DIR/requirements.txt" --quiet

echo "==> Copying skills to $SKILLS_DIR..."
mkdir -p "$SKILLS_DIR"
cp "$REPO_DIR/skills/"*.md "$SKILLS_DIR/"

echo "==> Registering MCP server with Claude Code..."
claude mcp add quant-agent -- python "$REPO_DIR/quant_agent/mcp_server.py"

echo ""
echo "Done! Restart Claude Code, then use /quant-agent or any technique skill."
echo "Run 'python demos/run_all_demos.py' to see all techniques on synthetic data."
