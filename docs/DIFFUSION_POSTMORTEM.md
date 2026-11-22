# Diffusion post-mortem (task 3.3)

Data: `data-full` VALID split (grouped split v1, 64 maps). Old training set is gone (D12), so absolute
numbers are not comparable to the report; the conditions are compared with each other.
Checkpoints: archived `2-new_arch/ep50_sp0.2_std0.005` (cold); archived `5b7efd0` encoder (latent).
Code: `archive/diffusion_7b9b37c/` (local; recoverable from commits `7b9b37c`, `5b7efd0`).

## Cold diffusion (ρ, valid, 64 maps; `results/postmortem_3_3/cold_causes_valid.json`)

| Condition | ρ | Verdict |
|---|---|---|
| input (x_T) | 0.794 | reference |
| direct, one decoder call at T | 0.976 | reference |
| Algorithm 2, matched mask and noise | 0.983 | reference |
| mask mismatch (sampler mask ≠ x_T mask) | 0.803 | **confirmed**: −0.18 |
| noise mismatch (sampler noise ≠ x_T noise) | 0.864 | **confirmed**: −0.12 |
| resample mask and noise every step | 0.766 | **confirmed**: destroys the gain |
| T sweep, K = 1 / 5 / 20 / 100 / 1000 | 0.982 / 0.983 / 0.983 / 0.983 / 0.983 | **rejected**: no error accumulation |

- The report's claim that Algorithm 2 is worse than its input does not reproduce on data-full: matched Algorithm 2 is the best of the three.
- The old sampler's default draws a fresh mask (`fixed_mask_rand=None`), so the old evaluations were probably run mismatched. That matches the largest effect here.
- **Network never sees the mask: confirmed for direct prediction, not the cause of the sampler failure.** Retraining with the observed-pixel mask as a 3rd input (2 seeds, same recipe as the random-t baseline) raises direct ρ from 0.953/0.956 to 0.972/0.976 and cuts MSE by 3–4×. With the realistic sampler (fresh randomness each step), Algorithm 2 stays at ρ ≈ 0.75–0.76 with or without the mask, so the iterative sampler is the bottleneck, not the input.

## Latent DDPM (`results/postmortem_3_3/latent_causes_valid.json`)

- **Decoder not conditioned on the observation: confirmed (code).** The decoder takes `z = Encoder(observation)` only, a 128-d summary, and never the spatial map.
- **InfoNCE and instance identity: partly supported.** Linear probe for class from z: 0.328 (majority chance 0.200). Nearest neighbour is the same class 19.8% of the time and the same cluster 8.1% (chance 1.4%). The latent carries class and instance information, but is not class-only.
- **Data scale vs unit-variance prior: confirmed.** Data std is 0.164. With the unit-variance prior's signal-to-noise scale, the signal is below noise at 74% of timesteps. With the actual data variance, it's below noise at 95%. Most of the schedule is wasted.

## Time comparison (D12; s = 0.7, σ = 0.005, 60 epochs, 2 seeds; `results/postmortem_3_3/time/`)

| Mode | ρ (mean) | seed spread | MSE (mean) |
|---|---|---|---|
| random t | 0.9509 | 0.0001 | 2.10e-4 |
| fixed t (= T) | 0.9513 | 0.0001 | 2.03e-4 |
| no time info | 0.9517 | 0.0003 | 2.02e-4 |

No difference beyond seed spread. Best validation losses differ (random 9e-5 vs 2e-4), but random-t losses are measured at easier, smaller t.

## Caveats
- One degradation draw on 64 valid maps for the cold and T-sweep runs. No seed spread yet.
- The checkpoint was trained on data no longer available, and the data differs from the report's.
