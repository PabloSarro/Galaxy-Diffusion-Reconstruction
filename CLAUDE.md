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
- `src/`: `model.py`, `dataset.py` (per-sample `groups` for grouped splits), `degradation.py`,
  `helpers.py`, `training.py` (resumable decoder loop), `train_decoder.py` (+ `.job`), `train_B1.job`,
  `splits.py` (grouped split), `eval_split.py` (per-class pixel), `eval_pixel.py`, `eval_args.py`,
  `baselines.py` (kNN, raw), `score_baselines.py`, `classify_split.py` (classifier), `physics_metrics.py`,
  `b1_closing.py`, `b1_classify.py`, `compare_logs.py`, `classification.py`, `visualise_imgs.py`,
  jobs `*.job`. `postmortem/`: `cold_causes_3_3.py`, `t_sweep_3_3.py`, `latent_causes_3_3.py`,
  `time_comparison_3_3.py` (+ `.job`).
- `tests/`: `test_units.py`, `test_resume.py`, `test_splits.py`, `test_eval_split.py`,
  `test_baselines.py`, `test_classify_split.py`, `golden/` (golden check incl. B1), `eval/`.
  Run: `envSummer1_conda/bin/python -m pytest tests` (about 5 min; golden loads data on CPU).
- `results/`: `Reconstruction/Model {1,2,3}` (old-split checkpoints + logs), `Classification/`,
  `splits/split_v1.json` (grouped split, seed 0, 70/15/15), `split_v1/` (3.1 outputs: baselines,
  classification, physics, diagnostics), `B1/` (3.2 seeds 1–2 at σ=0.22 and σ=0; `closing/` outputs),
  `postmortem_3_3/` (3.3 outputs), `smoke_2.2/`, `repro_1.1/` (local).
- `report/`: LaTeX sources (`shear_reconstruction.tex`, `LASTRO - Summer Project Report.tex`,
  `images/`); `report.zip` is the original bundle. Git-ignored.
- `archive/`: retired code, old results, old test scripts; `archive/diffusion_7b9b37c/` holds the old cold/latent diffusion code (git-ignored; see `archive/MOVED.md`).
- `envSummer1_conda/`: conda env used by the jobs (git-ignored). `data-full/`: data (read-only).
- Outside the repo: `/home/sarro/external/PyTorch-Network-Loader` (pin `0a12e38`).
- Keep this map current: update it whenever the structure changes.
- There's another project ongoing, sitting in the path ../SemProj/, hence if you see jobs pending/running that you didn't write, it must have been the other Claude Code session that I have on that.

## Starting a session
- Always inside screen (tmux is not installed on izar), so the session survives SSH
  disconnects and closing VS Code:
  `screen -DRS claude`, then `cd /home/sarro/SummerProj && claude`. Detach with `Ctrl-a d`.
- First read `TASKS.md`: "Status" and "Next session: start here".
- Anything long that you start (training, evaluation, sweeps) goes into a cluster job or its own
  named screen window, with its output logged to a file. Never a bare background process.
- Slurm: before every job, run `sshare -U -u $USER` and submit with the account that has the
  highest FairShare (accounts: cs-302, env540, math-454, master; partition gpu), but in case 
  'QOSMaxJobsPerUserLimit' is obtained, try the second with the highest FairShare, and so on.

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
- Whenever you want to delete something, either think again about it, or let me know about that
  and allow me to have the final decision.
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
