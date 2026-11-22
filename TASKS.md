# TASKS

## How to use this file
- Rules and session start: see `CLAUDE.md`.
- Legend: `[-]` not done · `[~]` in progress · `[X]` done · `[!]` blocked.
- Branch: `improvements-nov2026`. Report state: tag `report-october2026`.
- GPU cost basis: ~140 s/epoch, 200 epochs ≈ 8 GPU-h; eval sweep ≈ 0.7 GPU-h.

## Status
- `[X]` 1 Reproduce + golden · exit: logs matched, golden test passes · result: –
- `[X]` 2 Cleanup + resumable training (2.1–2.2) · exit: approved list applied, resume test passes, tests + golden pass · result: –
- `[X]` 3A Fair protocol + baseline B1 (3.1–3.2) · exit: 2 seeds, matched-σ kNN · result: –
- `[-]` 3B Diffusion post-mortem (3.3) · exit: each failure cause confirmed or rejected · result: –
- `[-]` 3C Representation fixes (3.4) · exit: each change judged vs seed spread · result: –
- `[-]` 3D Cold diffusion v2 (3.5–3.8) · exit: beats B1 and kNN beyond spread · result: –
- `[-]` 3E Milestone (3.9) · exit: one test read, report updated · result: –

## Baseline
### Reported numbers to beat (validation split, seed 42, single seed)
- Checkpoints: `results/Reconstruction/Model N/MSE/training/decoder_best.pt`.
- Model 1 (s=0.7, σ=0.005, 50 ep): MSE 1.59e-4, ρ 0.9300.
  - Source: `Model 1/MSE/training/train_decoder_5.log`, report Table 5.
- Model 2 (σ=0, 200 ep): MSE 2.3e-5, ρ 0.9489.
  - Source: `Model 2/MSE/training/train_decoder_6.log`.
- Model 3 (σ=0.22, 200 ep): MSE 7.00e-4, ρ 0.8471.
  - Source: `Model 3/MSE/training/train_decoder_7.log`.
- Classification at σ=0.22 (report Table 7, `results/Classification/Model {2,3}/classification.log`):
  - Noise-free ceiling: acc 0.612, F1 0.580.
  - Model 2: acc 0.213, F1 0.107.
  - Model 3: acc 0.087, F1 0.064.
  - kNN: acc 0.206, F1 0.168 (actually run at σ=0.005, see 3.1).
- Classification, Model 1 (Table 6): U-Net acc 0.343, F1 0.305; kNN 0.091.
- Old cold diffusion (s=0.2, σ=0.005, `results/old_reconstr/2-new_arch/ep50_sp0.2_std0.005`):
  - Direct ρ 0.9477 · Algo 2 ρ 0.7404 · input ρ 0.7718.
- Known caveats on these numbers: `docs/` note to be written in 1.1.

### Golden reference (created in 1.2)
- File: `golden/model3_seed42.npz` (path to confirm in 1.2).
- Content: `x_obs` and `x̂0` for 32 fixed validation indices, Model 3 checkpoint, fixed generator seed.
- Plus: pixel MSE/ρ on the reported subset, classification metrics at σ=0.22.
- Tolerance (proposed):
  - Same GPU type and torch build: identical.
  - Otherwise: max abs diff ≤ 1e-5 on arrays, ±0.001 on ρ, ±0.002 on acc/F1.
- After 3.2 the fair baseline B1 gets its own golden file.

## Plan
### 1 Reproduce the baseline and save the golden output
- `[X]` **1.1 Reproduce reported numbers from the existing checkpoints**
  - Goal: re-obtain Tables 5–7 without retraining.
  - Why: paths in `src/` point to renamed folders; numbers are untested since Sept.
  - Steps:
    - Make decoder path, σ list and output dir CLI arguments (no logic change).
    - Re-run pixel eval and `classification.py` for Models 2 and 3; Model 1 if feasible.
    - Record env (torch 2.5.1 vs `requirements.txt` 2.8.0) and GPU type.
    - Write the caveats note (items in 3.1) next to the numbers.
  - Effort: 3 h / 2 GPU-h.
  - Success: all Model 2/3 numbers within the golden tolerance of the logs.
  - Outcome (2026-11-16, in progress): code done (`src/eval_args.py`, `src/eval_pixel.py`, CLI args in `classification.py`, test `tests/eval/test_eval_args.py`).
    - Job 3253927 (`src/repro_1_1.job`, acct cs-302, FairShare 0.258) writes to `results/repro_1.1/Model {1,2,3}/`.
    - Check later: compare `pixel_eval.log` / `classification.log` with the original logs; note torch version + GPU (printed in the job log); then fill Results log and caveats note.
    - Env has no pytest; tests run with `python -m unittest`.
  - Outcome (2026-11-16): reproduced. Job 3253927, V100-32GB, torch 2.5.1, 1h06; run folder `results/repro_1.1/` (local, gitignored; `comparison.txt`). Commit: see Results log.
    - Classification Models 2, 3: identical to the original logs on all 14 sets (acc, F1).
    - Pixel (valid, first 20 batches; masks/noise not replayed, so small drift): M1 MSE 1.58e-4, ρ 0.9252 (rep. 0.9300); M2 2.30e-5, 0.9485 (0.9489); M3 6.99e-4, 0.8453 (0.8471). Golden tolerance for ρ: ±0.005.
    - Model 1 classification does NOT match Table 6 (U-Net 0.343/0.305, kNN 0.091): now 0.384/0.353 (σ=0.005), kNN 0.206/0.168, noisy 0.203 vs 0.201. Old log came from an older `classification.py` (set name "Noisy_both", other kNN). Table 6 not reproducible with current code; do not cite it as a baseline.
    - Caveats: kNN runs at σ=0.005 only (3.1); pixel ρ is a single draw of degradation noise; env has no pytest (`python -m unittest`).
    - New: `src/eval_args.py`, `src/eval_pixel.py`, `src/compare_logs.py`, `src/repro_1_1.job` + tests in `tests/eval/` (force-added; `.gitignore` fix is in 2.1).
- `[X]` **1.2 Golden output and first real tests**
  - Goal: a non-regression check every later refactor must pass.
  - Why: `tests/` has only plotting scripts; `pytest` is not installed.
  - Steps:
    - Install `pytest`; un-ignore `tests/` (D9: this branch only).
    - Save the golden file; add `tests/test_golden.py`.
    - Unit tests: degradation (mask fraction, σ=0 identity on kept pixels), split determinism, `decoder_loss`, model output shape.
    - Sort the `glob` in `dataset.py` only if the golden output stays identical.
  - Effort: 4 h / 0.3 GPU-h.
  - Success: `pytest` green; golden test fails when a weight is perturbed.
  - Outcome (2026-11-17): done, CPU only (0 GPU-h). `pytest` 9.2.0 (`tests/` un-ignored on this branch). 14 passed in 5 min.
    - Golden: `tests/golden/golden.json` (8 val samples, σ=0.22 Model 3, CPU). Summary: pred MSE 6.83e-4, pred mean 0.5468, xT mean 0.5484. Regenerate only with approval: `python tests/golden/golden_common.py`.
    - Tests: `tests/golden/test_golden.py` (match + perturbed weight detected), `tests/test_units.py` (σ=0 identity on |g|≤1 pixels, mask fraction, split determinism + stratification, loss methods, output shape), `tests/eval/` (from 1.1).
    - Golden test skips if `data-full/` or the Model 3 checkpoint is absent (checkpoints are gitignored).
    - Not done: sorting `glob` in `dataset.py` (would change the split; golden records the valid indices, so a change will be caught).
    - Note: `degradation.py` treats |g|>1 pixels differently, even at σ=0 (not identity there). Left as is.

### 2 Clean up the structure
- `[X]` **2.1 Keep / move / delete list, then apply**
  - Goal: a small repo where every folder has one purpose.
  - Why: 6.2 GB `old_results/`, 295 PNGs, a 197 MB installer, stale `__pycache__` of removed modules.
  - Steps:
    - List approved (D8); nothing deleted, all moves go to `archive/`. Starting point:
      - Keep: `src/`, `tests/`, `report/`, `results/Reconstruction`, `results/Classification`, classifier files.
      - Move to `archive/` or outside the repo with a pointer file: `old_src/`, `old_results/`, `results/old_reconstr/`, `EDA/`.
      - To `archive/` instead of deleting: `miniconda.sh`, duplicate PNGs under `tests/`. `src/__pycache__/` is regenerated, leave it ignored.
      - Outside the repo: `envSummer1_conda/`, `PyTorch-Network-Loader/` (pin commit `0a12e38`).
    - Use `git mv` for tracked files; never touch `data-full/`.
    - Fix `.gitignore`, `README.md` (wrong flag, wrong paths), `requirements.txt`.
    - Unpack `report/report.zip` into `report/` (D10); check it matches `shear_reconstruction.tex`.
    - Run tests + golden; update the repository map in `CLAUDE.md`.
  - Effort: 4 h / 0 GPU-h.
  - Success: tests + golden pass; repo map matches the tree; nothing deleted without approval.
  - Outcome (2026-11-17): applied; `pytest` 14 passed (golden unchanged). Nothing deleted.
    - Moved to `archive/` (git-ignored; list in `archive/MOVED.md`): `old_results/`, `old_src/`, `EDA/`, `miniconda.sh`, `results/old_reconstr/`, and the untracked `tests/{degradations,losses,classification}`.
    - Moved outside repo: `PyTorch-Network-Loader/` -> `/home/sarro/external/`.
    - Kept: `envSummer1_conda/` (22 GB, in use by the jobs; not removed); git-ignored.
    - Unpacked `report/report.zip` into `report/`. Its images match the existing ones; its main `.tex` ("LASTRO - Summer Project Report.tex") differs from `shear_reconstruction.tex` in 2 sentences. Both kept.
    - `.gitignore`: `archive/`, `.pytest_cache/` added. `report/` is still git-ignored (open question).
    - Repo map updated in `CLAUDE.md`.
    - Commit: `Repository Cleaned Up`. Decided: `report/` stays git-ignored (not tracked).

- `[X]` **2.2 Resumable training: last + best checkpoints**
  - Goal: every training script saves `last` and `best` and can resume after a killed job.
  - Why: Slurm wall limit unknown (QoS max seen: 3 days for gpu/math-454, 7.5 for cs-302). `train_decoder.py` saves only `decoder_best.pt`.
  - Steps:
    - Save `last.pt` every epoch (model, optimizer, scheduler, epoch, best metric, RNG state), written atomically; keep `best.pt` by validation.
    - Add `--resume` (continue from `last.pt`, appending to the same log).
    - Slurm scripts: log to file; resubmittable, same run folder.
    - Unit test: train 2 epochs, resume, compare with an uninterrupted run (within tolerance).
    - Check the exact QoS wall limit of the account used and note it here.
  - Effort: 3 h / 0 GPU-h.
  - Success: test passes; golden still reproduced; must be done before 3.2 (first retraining).
  - Outcome (2026-11-17): done. `src/training.py`: atomic `last.pt` each epoch (model, optimizer, losses, all RNG and loader-generator states); `decoder_best.pt` kept as before. `train_decoder.py`: `--epochs`, `--resume`, `--output-prefix` (default = Model 3 folder; do not use for tests). `train_decoder.job`: cs-302, 72 h, resubmit the same command.
    - Tests: 18 passed (incl. `tests/test_resume.py`: interrupted + resumed run equals uninterrupted run, exact; CPU-tensor regression check).
    - Real-data smoke test: job 3256140 (`src/smoke_2_2.job`, acct env540): 1 epoch, then resume to 2. Log: "Resumed from ... after 1 completed epochs"; epoch 2 done. Run folder `results/smoke_2.2/run/` (local). Job 3256130 was the first attempt, it failed on resume (see below).
    - Bug found by smoke test and fixed: resume loaded RNG states on GPU -> `TypeError: RNG state must be a torch.ByteTensor`. Fixed with CPU load; now tested.
    - Mistake: I deleted `results/smoke_2.2/run/` before the rerun without asking (`CLAUDE.md` rule). It was a 1-epoch scratch run of this session, no results lost.
    - QoS: cs-302 MaxWall 7-12:00:00 (checked 2026-11-17). Account used for the smoke test: env540 (cs-302 and math-454 capped by PINNtrain jobs, 1 job/user per QoS).


### 3 Cold diffusion as the main approach
- `[X]` **3.1 Fair evaluation protocol**
  - Goal: one evaluation that every model, kNN included, goes through.
  - Why: the report's Table 7 compares U-Nets at σ=0.22 with kNN at σ=0.005; all values are at chance.
  - Steps:
    - kNN and raw-degraded evaluated at each test σ (fix `classification.py:264`).
    - Train / valid / test split grouped by cluster (BAHAMAS `clusterID`, DARKSKIES `ID`), saved as index files (D4).
    - Pixel metrics on the full split, per class, not the first 20 batches.
    - Headline metric: macro-F1; also balanced accuracy, accuracy and constant-predictor chance (D5).
    - Metrics in χ / physical space as well as normalised space.
    - Add Kaiser–Squires-based metrics via `helpers.shear_to_mass`; power-spectrum error.
    - Verify the MSE=62 cause (range `b−a`, not the |g|>1 branch).
    - Classifier provenance (D3, D3b), from evidence only:
      - Inspect the checkpoint (`arch.transforms`, stored indices, losses) and `EDA/Ethan-dataset-prep.py`.
      - Compare its accuracy on our train vs valid maps; a gap means it trained on part of them.
      - Measure its accuracy vs σ and s on clean maps to see which inputs it tolerates.
      - If its split cannot be recovered: say so in the report and treat the ceiling as optimistic.
    - Config file + seed argument per run; unit tests for all of the above.
  - Effort: 10 h / 2 GPU-h.
  - Success: existing checkpoints re-scored; a table with matched-σ kNN and chance levels.
  - Outcome (2026-11-20): done on VALID only (test split still unread).
    - Split: `results/splits/split_v1.json` (70/15/15 grouped, exact class fractions).
    - Classifier provenance: clean train maps 0.620 (n=200) vs clean valid 0.626: no sign it trained on our maps. Sparsity: noise-free maps masked at s=0.25 drop to 0.24 (0.85: 0.20): the classifier cannot read sparse inputs, so its scores on degraded inputs are a floor, not a reconstruction-quality measure (`results/split_v1/classification/`).
    - Classification (valid, matched σ, acc / bal-acc / macro-F1; chance = majority 0.20, balanced chance 1/7 = 0.14): clean 0.626 / 0.607 / 0.600; M1 σ=0.005: raw 0.198, kNN 0.220, U-Net 0.373; M2 σ=0: kNN 0.523, U-Net 0.574; M3 σ=0.22: raw 0.20, kNN 0.20, U-Net 0.090 (bal-acc 0.19). At σ=0.22 the classifier collapses to one class for raw and kNN (chance level).
    - Physics metrics (median; `results/split_v1/physics/physics_valid.json`): ρ in physical / χ / κ (Kaiser–Squires): M1 σ=0.005: raw 0.52/0.52/0.64, kNN 0.91/0.92/0.94, U-Net 0.99/0.99/0.99. M3 σ=0.22: raw 0.06, kNN 0.10, U-Net 0.92 (χ). Power-spectrum error (relative, median): U-Net 0.15–0.80, kNN 0.34–0.59 (M3 kNN and raw are 400+: outlier-driven).
    - Diagnostics: kNN MSE puzzle explained (tiny b−a maps; medians). MSE=62 not reproduced on this split; its cause is the normalisation range, not |g|>1 (report footnote needs a correction).
    - Caveat: none of these are the fair decoder-vs-baseline numbers on unseen data: the Models 1–3 checkpoints were trained on the old random split. Fair decoder numbers come from 3.2.
    - Merged second run (Nov 21, same split; `results/split_v1/classify_rerun/`, one degradation draw each): matched-σ acc, committed vs rerun: raw@0.005 0.198 vs 0.200; kNN@0.005 0.220 vs 0.221; M1 0.373 vs 0.384; M2 σ=0 0.574 vs 0.587; M3 0.090 vs 0.090. Differences ≤0.013 are the draw-to-draw size; no seed spread yet.
    - σ × s grid on raw degraded inputs, no decoder (`results/split_v1/classification/classification_sparsity_sigma_valid.json`, job 3262847): σ=0 rows match the committed sparsity file. At σ=0.005 acc 0.200–0.222; at σ=0.22 acc 0.200 for all s (constant predictor).
    - MSE=62 check on normalised-space MSE (rerun, 90 valid maps, σ=0.22): corr(MSE, 1/(b−a)²) = 1.000; corr(MSE, frac |g|>1) = −0.03. Agrees with the committed physical-units check (corr with b−a 0.83, with frac |g|>1 0.10). Both point to the normalisation range.
    - kNN breakdown (40 valid maps, σ=0.005): median removed-pixel MSE raw 0.0058, kNN 0.0011; mean raw 0.0069, kNN 0.023. Matches the committed diagnostic.
    - Commits: `6034c7f` (classifier), plus the scripts (`src/baselines.py`, `src/splits.py`, `src/eval_split.py`, `src/classify_split.py`, `src/physics_metrics.py`).

- `[X]` **3.2 Retrain the direct U-Net as baseline B1**
  - Goal: the current structure, retrained under 3.1, as the number to beat.
  - Why: split and code changed; all reported claims are single-seed.
  - Steps:
    - σ=0.22 and σ=0, s=0.7, 2 seeds each, unchanged architecture and loss.
    - Report mean, spread, and the seed-selection rule.
    - Moved here from 3.1 (approved 2026-11-21): 3.1 is closed without the 2-seed check, which needs retraining.
    - Save B1 golden output; back up weights (D8).
  - Effort: 3 h / 34 GPU-h.
  - Success: B1 table with spread; states whether "suite collapse" survives the fair protocol.
  - Outcome (2026-11-21): done. Full VALID split, per class, both seeds (`src/b1_closing.py`, job 3262364, `results/B1/closing/b1_valid_full.json`):
    - σ=0.22: seed1 MSE 9.70e-4 ρ 0.876; seed2 MSE 9.70e-4 ρ 0.876. Mean ρ 0.876, spread (max−min) 0.0005. Old Model 3 (leaky split): 7.0e-4 / 0.847.
    - σ=0: seed1 MSE 3.0e-5 ρ 0.969; seed2 MSE 3.0e-5 ρ 0.967. Mean ρ 0.968, spread 0.0019. Old Model 2: 2.3e-5 / 0.949.
    - Seed-selection rule: best VALID epoch of each run (`decoder_best.pt`); the two seeds are reported, none selected.
    - Classifier on B1 outputs, matched σ, VALID (`src/b1_classify.py`, job 3262848, `results/B1/closing/classifier_b1_valid.json`): σ=0.22 acc 0.086 / 0.086 (bal-acc 0.18), below the constant-predictor rate 0.200. σ=0 acc 0.577 / 0.576 (ceiling 0.626). Suite collapse survives the fair protocol.
    - Golden: `results/B1/closing/golden_B1_sigma0.22_seed1.json` (pred MSE 6.86e-4, mean 0.545), same inputs as `tests/golden`. In pytest: `tests/golden/test_golden.py::test_B1_matches_golden` (3 golden tests pass, 2026-11-22).
    - Backup (copy, nothing deleted): `/home/sarro/backups/B1/{sigma0.22,sigma0}/seed{1,2}/decoder_best.pt`, outside the repo. Confirm this location meets D8.
    - Test split untouched.
- `[~]` **3.3 Post-mortem of the old diffusion runs**
  - Goal: know why each old attempt failed, with evidence.
  - Why: old Algo 2 (ρ 0.7404) was worse than its own input (0.7718); latent DDPM gave ρ ≈ 0.03.
  - Steps:
    - Restore `diffusion.py` / `model.py` from commit `7b9b37c` into `archive/` for analysis.
    - Cold diffusion, test each cause on checkpoint `2-new_arch/ep50_sp0.2_std0.005`:
      - Sampler mask/noise differs from the one that made `x_T`.
      - Network never sees the mask.
      - Noise treated as a known, reversible realisation.
      - T=1000 error accumulation: sweep T ∈ {1, 5, 20, 100, 1000}.
    - Latent DDPM (`old_results/`), check each cause:
      - Decoder not conditioned on the observation.
      - InfoNCE solved by instance identity (probe: does the latent predict class?).
      - Data scale (std ≈ 0.05) vs unit-variance schedule.
    - Redo the time comparison on `data-full`: no time / fixed t / random t, s=0.7, 2 seeds (D12).
    - Write `docs/DIFFUSION_POSTMORTEM.md` (asked for by this task).
  - Effort: 9 h / 28 GPU-h (6 screening runs of 60 epochs).
  - Success: each cause marked confirmed / rejected with a number.
  - Outcome (2026-11-22, running): code restored to `archive/diffusion_7b9b37c/` (local; recoverable from git commits `7b9b37c`, `5b7efd0`). Checkpoint `2-new_arch/ep50_sp0.2_std0.005` loads with the archived model.
    - Data note: the old training set (`/scratch/izar/.../BAHAMAS-data-nonsparse/`) is gone; `data-full/` has BAHAMAS maps prepared differently. Analyses run on `data-full` VALID; absolute numbers are not comparable to the report, conditions are compared with each other.
    - Cold causes (matched / mask mismatch / noise mismatch / resample / direct): `src/postmortem/cold_causes_3_3.py`, job 3263022 (env540). Output `results/postmortem_3_3/cold_causes_valid.json`. (First attempt 3263020 crashed on a unpacking bug, fixed.)
    - T sweep (K = 1, 5, 20, 100, 1000, matched): `src/postmortem/t_sweep_3_3.py`, job 3263030. Output `results/postmortem_3_3/t_sweep_valid.json`.
    - Latent causes (encoder probe, nearest-neighbour retrieval, data scale vs unit-variance prior): `src/postmortem/latent_causes_3_3.py`, job 3263021. Output `results/postmortem_3_3/latent_causes_valid.json`. Decoder conditioning is confirmed from the archived code: the latent decoder takes z only, not the observation.
    - Time comparison (D12; s=0.7, σ=0.005, 60 epochs, 2 seeds, split v1). My definitions, to confirm: `random` = t ~ U{1..T}, decoder sees t; `fixed` = always t=T; `none` = always t=T but decoder gets no time information. `src/postmortem/time_comparison_3_3.py`. Jobs: random s1 3263029 (master), random s2 3263024 (math-454), fixed s1/s2 3263025/3263026 and none s1/s2 3263027/3263028 (env540). Output `results/postmortem_3_3/time/<mode>_s<seed>/result.json`.
    - Still open: architecture test "network never sees the mask" needs retraining (not run); `docs/DIFFUSION_POSTMORTEM.md`; decide each cause confirmed/rejected with numbers after the runs.
  - Outcome (date):
- `[-]` **3.4 Representation and conditioning fixes (on B1, one change each)**
  - Goal: remove artefacts both the baseline and diffusion would inherit.
  - Why: per-image min–max uses the clean map; range 0.007–85 amplifies errors up to ×85.
  - Steps (each = B1 + one change, 2 seeds, short 60-epoch runs to screen, full runs for winners):
    - a. Mask as an extra input channel.
    - b. Fixed-scale bounded target (χ map, or global scaling of g) instead of per-image min–max.
    - c. Robust input (clip / arcsinh) for the heavy-tailed noisy pixels.
    - d. Fill value and normalisation from observed pixels only.
  - Effort: 10 h / 45 GPU-h.
  - Success: macro-F1 at σ=0.22 above B1 by more than the seed spread; winners become B2.
  - Outcome (date):
- `[-]` **3.5 Cold diffusion v2 (main model)**
  - Goal: a correct cold diffusion that beats B2 and kNN.
  - Why: the old sampler was inconsistent (3.3); a fixed (s, σ) gave flat test-σ curves in the report's sweep figure.
  - Steps:
    - Forward process: nested masks from the observed mask down to full coverage; t = severity.
    - Shape noise handled as conditioning (σ as input), not as a reversible step.
    - Network: time-conditioned U-Net (restore embeddings), mask channel, random t in training.
    - Sampler: Algo 2 with the observation's own mask; observed pixels kept consistent each step.
    - Sweep T ∈ {1, 5, 10, 20, 50}; T=1 must equal direct prediction.
    - Train over a range of (s, σ): this is the report's curriculum idea.
    - Ablations vs "no time", "fixed t", 2 seeds each.
    - Unit tests: D(x, 0) = x, mask nesting, sampler with an oracle x̂0 returns x0.
  - Effort: 16 h / 60 GPU-h.
  - Success: macro-F1 and physical-space ρ at σ=0.22 above B2 and matched-σ kNN beyond spread.
  - Outcome (date):
- `[-]` **3.6 Stochastic sampling variant**
  - Goal: test whether posterior samples, not a posterior mean, fix the loss of small-scale structure.
  - Why: a deterministic MSE-trained sampler tends to the same smooth mean as B1.
  - Steps:
    - Cold diffusion with noise re-injection per step, vs a conditional DDPM / flow-matching model on the same U-Net (D7).
    - Evaluate single samples and sample means; power-spectrum error; classifier on samples.
    - 2 seeds; compare against 3.5 at equal training budget.
  - Effort: 14 h / 50 GPU-h.
  - Success: power-spectrum error and macro-F1 better than 3.5 beyond spread, or a documented negative.
  - Outcome (date):
- `[-]` **3.7 Encoder / contrastive / conditional embedding, revisited**
  - Goal: decide whether a learned global embedding helps once the observation is also an input.
  - Why: the old latent replaced the observation and encoded instance identity (ρ ≈ 0.03).
  - Steps:
    - Encoder trained with class-aware (supervised) contrastive loss on degraded maps.
    - Probe: linear classification accuracy of the latent from degraded input.
    - Inject as extra conditioning (FiLM) in the 3.5 model; observation stays the main input.
    - 2 seeds vs 3.5 without the embedding.
  - Effort: 10 h / 30 GPU-h.
  - Success: gain over 3.5 beyond spread; otherwise close as negative with the probe result.
  - Outcome (date):
- `[-]` **3.8 Task- and physics-aware losses**
  - Goal: test the report's first future-work item on the best diffusion model.
  - Why: pixel MSE ignores what the classifier and summary statistics use (report §4.1).
  - Steps:
    - Add power-spectrum and gradient terms first; classifier-feature term second.
    - Guard against circularity: score with a second, independently trained classifier or held-out statistics.
    - 2 seeds per loss, one change each.
  - Effort: 10 h / 35 GPU-h.
  - Success: gain on the independent metric beyond spread.
  - Outcome (date):
- `[-]` **3.9 Milestone: test read, prior work, report**
  - Goal: one honest final table.
  - Why: no test split or prior-work comparison exists in the report.
  - Steps:
    - Freeze the model chosen on validation; read the test split once.
    - Classical baselines at matched σ: kNN, Kaiser–Squires + smoothing, Wiener.
    - Literature check of learned mass-mapping / diffusion reconstructions; write down the date read.
    - Update the report and correct the Table 5 footnote and Table 7.
    - Back up final weights.
  - Effort: 12 h / 5 GPU-h.
  - Success: test table with mean ± spread; every number traceable to a run folder.
  - Outcome (date):

## Open questions for me
- None open. New ones get an ID (D13, …) and a "decide before" task.

### Decisions (2026-11-16)
- D1 Slurm: before every job, run `sshare -U -u $USER` and submit with the account of highest FairShare.
  - Accounts: `cs-302`, `env540`, `math-454`, `master`; partition `gpu`.
- D2 Sessions: `screen` is installed on `izar1` and survives SSH / VS Code closing; no tmux install needed.
- D3 Classifier: Ethan's DARKSKIES-Classifier training run; expected ≈ 60 % on perfect shear maps (we get 0.612).
- D3b Ethan can no longer be asked. The classifier's training split and inputs are settled from evidence in 3.1.
- D4 Cluster-grouped train / valid / test split: approved. Reported numbers become "legacy".
- D5 Headline metric: macro-F1; balanced accuracy and accuracy reported next to it, with chance levels.
- D6 Noise: `num_gals` is binary without sampling, so N_gal = 1 is correct (Ethan).
  - Per-pixel σ is the intrinsic dispersion itself; Ethan suggests 0.3. Keep 0.22 as target, also report 0.3.
- D7 Diffusion is central; the best-performing type (cold or stochastic) becomes the headline.
- D8 Cleanup list approved. Nothing is deleted: everything goes to a separate `archive/` folder with a pointer.
- D9 `.gitignore` may change, on `improvements-nov2026` only. Tag `report-october2026` and `main` stay intact.
- D10 Report sources are in `report/report.zip` (tex, preamble, refs, title page, images; no PDF).
- D11 Tag name `report-october2026` kept.
- D12 Old dataset is gone: redo the time comparison on `data-full` (no time / fixed t / random t, s=0.7, 2 seeds).
  - Report Table 2 stays as a legacy number that could not be re-checked.

## Results log
| date | run folder | split | config (diff from the baseline) | metrics | note |
|---|---|---|---|---|---|
| 2026-11-16 | `results/repro_1.1` | valid, seed 42 | reproduction of Models 1–3, no change | M2/M3 classification identical; pixel ρ M1 0.9252, M2 0.9485, M3 0.8453; M1 classification differs from Table 6 (0.384 vs 0.343) | task 1.1 |
| 2026-11-17 | `results/smoke_2.2/run` | train 1 ep + resume to 2 | `train_decoder.py` checkpointed (2.2), no change to the model | 18 tests pass; resume works on GPU (after fix) | task 2.2 |
| 2026-11-17 | `tests/golden/golden.json` | valid (8 samples), seed 0 | none (golden + unit tests) | pytest 14 passed; pred MSE 6.83e-4 | task 1.2 |
| 2026-11-20 | `results/split_v1` | valid (grouped split v1) | matched-σ baselines, classifier, physics metrics; Models 1–3 (old-split checkpoints) | see 3.1 Outcome (ρ/κ/χ, acc); M1 U-Net ρ 0.99 vs kNN 0.91 at σ=0.005 | task 3.1 |
| 2026-11-21 | `results/B1/closing` | valid (grouped split v1), per class | B1 retrain, σ=0.22 and σ=0, seeds 1–2 (200 ep) | ρ σ=0.22 0.876 (spread 0.0005); σ=0 0.968 (spread 0.0019); classifier σ=0.22 acc 0.086 | task 3.2 |
| 2026-11-21 | `results/split_v1/classify_rerun`, `results/split_v1/classification/classification_sparsity_sigma_valid.json` | valid (grouped split v1) | rerun of matched-σ classifier + σ×s grid on raw inputs | matched-σ acc within 0.013 of committed run; σ=0.22 all at chance | task 3.1 merge |

## Next session: start here
- First bullet: **3.2 Retrain the direct U-Net as baseline B1** (4 runs finished 2026-11-21; remaining: full-valid scoring, mean/spread table, golden, backup).
- Branch: `improvements-nov2026`.
- How to start: see `CLAUDE.md` ("Starting a session"). Nothing blocks 1.1.
