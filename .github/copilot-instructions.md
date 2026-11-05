Read and follow the guidelines in [AGENTS.md](../AGENTS.md).

## Shell scripting standards

Assume this codebase will be maintained using Copilot and VS Code. Favor
structures that are resilient to automated edits.

- Never use heredocs unless explicitly requested or no practical alternative exists.
- Generate configuration artifacts as separate files when feasible.
- Prefer templates, structured generators (`jq`, `yq`, Python), or `printf` over
  embedded multi-line shell text.
- Optimize generated shell for maintainability under AI-assisted editing.
- Avoid constructs that depend on matching opening and closing delimiters.
- When generating Bash, assume future edits will be performed by Copilot.

Preferred order for generating multi-line content:

1. Separate files
2. Templates
3. Structured generators (`jq`, `yq`, Python)
4. `printf`
5. Heredocs only when no practical alternative exists
