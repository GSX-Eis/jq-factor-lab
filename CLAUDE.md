# JQ Factor Lab for Claude Code

Start with [AGENTS.md](AGENTS.md). To load the repository as a local plugin, run
`claude --plugin-dir .` from the repository root, then use
`/jq-factor-lab:factor-research`. The [Factor Research Skill](skills/factor-research/SKILL.md)
covers the research loop, CLI or MCP tool choice, and statistical gates.

Capture stdout and stderr separately, check the process exit code, and verify
the current `summary.json` before using metrics. Start with synthetic data.
Preserve point-in-time signals, execution lag, calendar coverage, actual label
exits, and all TrialLedger attempts.
