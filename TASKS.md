# TASKS

## How to use this file
- Rules and session start: see `CLAUDE.md`.
- Legend: `[-]` not done · `[~]` in progress · `[X]` done · `[!]` blocked.
- Branch: `improvements-nov2026`. Report state: tag `report-october2026`.
- GPU cost basis: ~140 s/epoch, 200 epochs ≈ 8 GPU-h; eval sweep ≈ 0.7 GPU-h.

## Status
- `[-]` 1 Reproduce + golden · exit: logs matched, golden test passes · result: –
- `[-]` 2 Cleanup · exit: approved list applied, tests + golden pass · result: –
- `[-]` 3A Fair protocol + baseline B1 (3.1–3.2) · exit: 2 seeds, matched-σ kNN · result: –
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
- `[-]` **1.1 Reproduce reported numbers from the existing checkpoints**
  - Goal: re-obtain Tables 5–7 without retraining.
  - Why: paths in `src/` point to renamed folders; numbers are untested since Sept.
  - Steps:
    - Make decoder path, σ list and output dir CLI arguments (no logic change).
    - Re-run pixel eval and `classification.py` for Models 2 and 3; Model 1 if feasible.
    - Record env (torch 2.5.1 vs `requirements.txt` 2.8.0) and GPU type.
    - Write the caveats note (items in 3.1) next to the numbers.
  - Effort: 3 h / 2 GPU-h.
  - Success: all Model 2/3 numbers within the golden tolerance of the logs.
  - Outcome (date):
- `[-]` **1.2 Golden output and first real tests**
  - Goal: a non-regression check every later refactor must pass.
  - Why: `tests/` has only plotting scripts; `pytest` is not installed.
  - Steps:
    - Install `pytest`; un-ignore `tests/` (D9: this branch only).
    - Save the golden file; add `tests/test_golden.py`.
    - Unit tests: degradation (mask fraction, σ=0 identity on kept pixels), split determinism, `decoder_loss`, model output shape.
    - Sort the `glob` in `dataset.py` only if the golden output stays identical.
  - Effort: 4 h / 0.3 GPU-h.
  - Success: `pytest` green; golden test fails when a weight is perturbed.
  - Outcome (date):

### 2 Clean up the structure
- `[-]` **2.1 Keep / move / delete list, then apply**
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
  - Outcome (date):

### 3 Cold diffusion as the main approach
- `[-]` **3.1 Fair evaluation protocol**
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
    - Classifier provenance: Ethan's run (D3); ask him about its split and inputs (D3b).
    - Config file + seed argument per run; unit tests for all of the above.
  - Effort: 10 h / 2 GPU-h.
  - Success: existing checkpoints re-scored; a table with matched-σ kNN and chance levels.
  - Outcome (date):
- `[-]` **3.2 Retrain the direct U-Net as baseline B1**
  - Goal: the current structure, retrained under 3.1, as the number to beat.
  - Why: split and code changed; all reported claims are single-seed.
  - Steps:
    - σ=0.22 and σ=0, s=0.7, 2 seeds each, unchanged architecture and loss.
    - Report mean, spread, and the seed-selection rule.
    - Save B1 golden output; back up weights (D8).
  - Effort: 3 h / 34 GPU-h.
  - Success: B1 table with spread; states whether "suite collapse" survives the fair protocol.
  - Outcome (date):
- `[-]` **3.3 Post-mortem of the old diffusion runs**
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
    - Redo the time ablation on `data-full` (D12).
    - Write `docs/DIFFUSION_POSTMORTEM.md` (asked for by this task).
  - Effort: 8 h / 4 GPU-h.
  - Success: each cause marked confirmed / rejected with a number.
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
| ID | Question | Decide before |
|---|---|---|
| D12 | Report Table 2 (fixed-time vs no-time U-Net) was run on `/scratch/izar/sarro/BAHAMAS-data-nonsparse`, which is gone. Do you still have that dataset, or do I redo the comparison on `data-full`? | 3.3 |
| D3b | Ask Ethan: which maps / split trained `network_v8_1.pth`, and did it ever see noisy or masked inputs? | 3.1 |

### Decisions (2026-11-16)
- D1 Slurm: before every job, run `sshare -U -u $USER` and submit with the account of highest FairShare.
  - Accounts: `cs-302`, `env540`, `math-454`, `master`; partition `gpu`.
- D2 Sessions: `screen` is installed on `izar1` and survives SSH / VS Code closing; no tmux install needed.
- D3 Classifier: Ethan's DARKSKIES-Classifier training run; expected ≈ 60 % on perfect shear maps (we get 0.612).
- D4 Cluster-grouped train / valid / test split: approved. Reported numbers become "legacy".
- D5 Headline metric: macro-F1; balanced accuracy and accuracy reported next to it, with chance levels.
- D6 Noise: `num_gals` is binary without sampling, so N_gal = 1 is correct (Ethan).
  - Per-pixel σ is the intrinsic dispersion itself; Ethan suggests 0.3. Keep 0.22 as target, also report 0.3.
- D7 Diffusion is central; the best-performing type (cold or stochastic) becomes the headline.
- D8 Cleanup list approved. Nothing is deleted: everything goes to a separate `archive/` folder with a pointer.
- D9 `.gitignore` may change, on `improvements-nov2026` only. Tag `report-october2026` and `main` stay intact.
- D10 Report sources are in `report/report.zip` (tex, preamble, refs, title page, images; no PDF).
- D11 Tag name `report-october2026` kept.

## Results log
| date | run folder | split | config (diff from the baseline) | metrics | note |
|---|---|---|---|---|---|

## Next session: start here
- First bullet: **1.1 Reproduce reported numbers from the existing checkpoints**.
- Branch: `improvements-nov2026`.
- How to start: see `CLAUDE.md` ("Starting a session"). Nothing blocks 1.1.
