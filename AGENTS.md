# Project Instructions for AI Agents

`FINAL_PROJECT_BLUEPRINT.md` is the project's **living specification and single source of truth**. Read it fully
before making any project-level change or starting a new phase.

## Standing Rule 1 — Auto-update the living spec

Whenever any project-level change occurs or is discovered (source, dataset strategy, architecture, phases,
preprocessing, model, evaluation, requirements), automatically update `FINAL_PROJECT_BLUEPRINT.md` to reflect the
new final decision and append an entry to its Change Log with the reason. Never change requirements silently. Do not
wait for the user to remind you.

## Standing Rule 2 — Git/GitHub workflow (full text in Blueprint §10)

- GitHub = project history + backup. Prefer: work → verify → document → Git checkpoint → continue.
- At meaningful checkpoints (phase completion, major verified implementation, important validation, before a major
  change) explicitly remind the user: "Git checkpoint reached..." and suggest a concise commit message. Do not
  interrupt trivial edits.
- **Never** run `git commit` or `git push` automatically — only when the user explicitly asks.
- Before starting a new major phase: verify the previous phase is complete per the blueprint and that
  implementation + documentation are consistent; then recommend a checkpoint if warranted.
- Do not preserve mistakes to keep Git history clean. If earlier work is wrong/incomplete, stop, assess, correct it,
  and record the change in the blueprint + Change Log.
- Before recommending a major push: check `.gitignore` covers datasets/secrets/temp files, important source + docs
  are included, and the repo is understandable on GitHub.
- Never fabricate commits, results, metrics, experiments, or history.

## Environment notes

- OS: Windows (win32), shell: PowerShell 5.1. Prefer full PowerShell syntax over POSIX-only constructs.
- `.venv/` exists and is gitignored; use `python` from it or the system interpreter as appropriate.