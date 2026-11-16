# CLAUDE.md

These are standing rules for every Claude Code session in this repo. Planning, progress and
results live in `TASKS.md`. This file only says how to work.

## Project
- Galaxy-Diffusion-Reconstruction.
- Reported results: "report/". Every improvement is measured against its baseline
  (`TASKS.md` → Baseline).
- Setup: README.md.

## Repository map
"src/" features the model, training, evaluation, data loading, and more source files, with results the outcomes.
- Data: "data-full/". Read-only: never write, move or delete anything there.
- Keep this map current: update it whenever the structure changes (e.g. after the cleanup task).

## Starting a session
- Always inside tmux, so the session survives SSH disconnects:
  `tmux new -As claude`, then `cd <<REPO_PATH>> && claude`.
- First read `TASKS.md`: "Status" and "Next session: start here".
- Anything long that you start (training, evaluation, sweeps) goes into a cluster job or its own
  named tmux window, with its output logged to a file. Never a bare background process.

## TASKS.md is the single source of truth
- A fresh session must see from it alone everything that was done, every outcome, and what's
  next. That includes ablations, small tests used to pick between configs, fine-tunings, and
  failed or negative runs.
- Legend: `[-]` not done · `[~]` in progress · `[X]` done · `[!]` blocked.
- When you finish or change a task, fill its "Outcome (date):" line (result, run folder, commit)
  and add a row to the Results log.
- Keep lines short. When a section grows long, move finished details to
  `docs/TASKS_HISTORY.md` and leave a one-line summary.
- Add any new task you find as a bullet in the right section, with its reason.

## Rules for experiments
- Decide on a validation split under test-like conditions. Read the test number once per
  milestone, never to choose between configs.
- At least 2 seeds per config. Compare against the seed spread; a gain smaller than the spread
  is not a gain. Report the mean, the spread, and which seed was picked by which rule.
- An ablation is the baseline plus one change. Retrain the baseline whenever training-relevant
  code changes.
- Every refactor must pass the tests and reproduce the golden baseline output (identical, or
  within the tolerance stated in `TASKS.md`).
- Every new script or module gets a unit test.
- Before claiming a position against prior work, check the current papers / leaderboards and
  write down the date you read them.
- Never delete run folders, checkpoints or data without asking me. Keep final weights
  somewhere backed up.

## Git
- Commits are authored by my configured git user only. No "Co-Authored-By", no "Generated
  with", no mention of Claude or AI in any commit message, trailer, tag or PR text. If a setting
  or hook would add one, tell me instead of committing.
- Commit messages: 2–4 words, Each Word Capitalised (e.g. `Baseline Reproduced`).
- Work on the current improvement branch (see `TASKS.md`). Push to `origin` (my fork) only,
  never to `upstream`. Never merge into `main` without asking me.

## Way of working
- One bullet of `TASKS.md` at a time, the one I name.
  1. Implement it, then run the tests and the golden check.
  2. Launch the needed runs.
  3. Update `TASKS.md` (status, outcome, results log).
  4. Commit and push. If a run is still going, commit with `[~]` and note what to check later.
  5. STOP and report:
     - What changed.
     - Test / golden results.
     - Run names and locations.
     - The commit hash.
     - Your proposed next bullet.
- Wait for my confirmation before the next bullet, unless I told you in advance to do the next
  X bullets on your own.
- If a decision can't be undone (deleting files, rewriting history, merging), always ask
  first, even inside an "X bullets" batch.
