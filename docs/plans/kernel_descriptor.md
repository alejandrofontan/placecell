# Kernel descriptor — a learned image descriptor whose cosine is the shared-information kernel

- **Author:** Alejandro Fontan (planned with Claude, Opus 5.5)
- **Created:** 2026-09-30
- **Status:** design agreed in discussion, nothing implemented; five points still open (see the last section)
- **Pinned to:** placecell `6bc3d30` (`main`), AllFeature `8e03732`

## Goal

Train an image descriptor `x = f(I)` such that `⟨x_i, x_j⟩ ≈ K_ij`, where `K` is the shared-information kernel we build from a COLMAP reconstruction (`tools/colmap_information_kernel.py`), so placecell's online insertion and culling decisions run on a kernel calibrated as information rather than on MegaLoc's retrieval cosine. MegaLoc's retrieval descriptor stays exactly as it is for relocalization and loop candidates.

## Why MegaLoc's cosine is not already that kernel

- **Architecture (SALAD, Izquierdo & Civera, CVPR 2024).** DINOv2-B patch tokens (23×23 = 529 at 322 px) are soft-assigned to 64 clusters by Sinkhorn with a dustbin; the marginals force every cluster to take mass exactly 1 (the dustbin takes n − m ≈ 465), each cluster vector is L2-normalised on its own, concatenated with the normalised global token and normalised again. Before MegaLoc's final projection (16640 → 8448) the cosine is exactly the average of 65 block cosines, and every block counts equally whatever evidence it aggregated. Likely source of the ~0.37 common-mode floor we centre away (hypothesis, untested).
- **Loss (multi-similarity, Wang et al. 2019).** SALAD uses α = 1, β = 50, λ = 0 with an MS miner at ε = 0.1; MegaLoc sums one MS loss per sub-batch (six), hyperparameters unpublished. The loss ranks and has no absolute level: a negative more than ε below the weakest positive gets no gradient, so unrelated pairs settle anywhere (the floor), and all positives are pulled alike, so nothing orders views within a place — exactly the structure the gram-greedy culler reads.
- **Labels.** Place classes, binary. SALAD: GSV-Cities only (GPS + heading). MegaLoc: 4 of 6 sub-batches GPS-defined (GSV-Cities, SF-XL ×2, MSLS), 2 geometry-defined (MegaScenes: ≥ 1 % shared 3D points; ScanNet: < 10 m and < 30°). Visual overlap is supervised only as a threshold, never as a graded quantity.

## What it would give placecell

A calibrated cosine is ≈ 0 for disjoint views: no centring (`centred = false`, as in item mode), no issue-#5 degeneracy, and the online VPR kernel on the same tau scale as the COLMAP kernel.

## Why it should be learnable

- Unit-norm descriptors can only produce a PSD, unit-diagonal kernel, so the target must be one too. The item cosine is exactly the cosine of each view's indicator vector over 3D points: the ideal descriptor is a compressed, scene-agnostic version of "which 3D content this image sees", which is structurally what SALAD already computes (soft-assign patches, aggregate, normalise).
- Dimension is not a limit: exact zeros between many mutually disjoint places need near-orthogonal vectors, and 8448 dimensions hold astronomically many at ε ≈ 0.05.

## Facts the implementation needs

- **MegaLoc:** DINOv2-B + SALAD (64 clusters × 256 + 256-d token = 16640) → linear 16640 → 8448 → L2 norm, 228 M parameters. Trained at 224×224, inferred at 322×322 (= 23·14). Only the last 4 DINOv2 blocks, SALAD and the projection were trained. One `backward()` per sub-batch before a single `step()` (their Algorithm 1): 360 → 60 GB, one A100 — reuse it for multi-trajectory batches.
- **Preprocessing:** the README applies `ToTensor → Normalize(ImageNet) → Resize([322, 322], antialias=True)`; `MegaLoc.forward` resizes (antialias) only when a side is not a multiple of 14. Training and inference must do the same, including TartanAir's 640×480 → 322×322 squash. If our TensorRT path ever drifts from PyTorch, check its resize against this (`megaloc_test --reference`).
- **SALAD code** (`models/aggregators/salad.py`): 3 Sinkhorn iterations, learnable scalar dustbin, score and cluster MLPs are 1×1 convs with 512 hidden units and dropout 0.3, token MLP 512 hidden. The dustbin ablation says it absorbs sky, road and dynamic objects (MSLS-val R@1 92.2 → 91.4 without it).
- **COLMAP:** the GT-overlap pair list goes in through COLMAP's custom-pairs matching (a text file of image-name pairs); only those pairs are matched and verified.

## Known risks

- **Low-texture frames** (walls, blur): every cluster still takes mass 1 and is normalised, so the descriptor is normalised noise plus generic content. Watch these in the validation, and they motivate ablation 8.3–8.4.
- **Forward-facing driving footage** is MegaLoc's own named weak case (CliqueMining beats it on MSLS); also forests and caves (AnyLoc better). `seasonsforest` in validation partly covers the latter.
- **Synthetic-to-real gap:** TartanAir is rendered; the frozen backbone and the real-sequence test (decision 13) are the guards.
- **GPS label noise** in MegaLoc's own training data (one of its four failure categories) — part of why its retrieval cosine is a poor information proxy, not something we inherit in the new branch.

## Design decisions

### Data — TartanAir-train (`tartanair-train` in VSLAM-LAB, 369 trajectories, 18 environments, 640×480, fx = fy = cx = 320, cy = 240, GT poses + depth)

1. **One model and one kernel per trajectory.** No cross-trajectory models.
2. **Fixed stride 5** (every 5th frame, 6 Hz). Keeps the near-duplicate pairs (K ≈ 1) the culler must learn from; ~25× cheaper than all frames.
3. **Candidate pairs from GT frustum overlap**, not exhaustive, sequential or retrieval matching (a missed pair is a false zero in the target; MegaLoc retrieval would teach the model MegaLoc's blind spots). GT depth subsampled every 8th pixel, reprojected i → j and j → i; a point counts when it lands inside the image and agrees with j's depth within 5 % (occlusion); the pair goes to COLMAP when the smaller of the two fractions is ≥ 5 %. COLMAP extracts and matches only those pairs.
4. **GT model** with `tools/colmap_gt_model.py` (COLMAP's verified matches + GT poses and depths). Its verification step checks the NED → camera convention (`ned_pose_to_camera` in `dataset_tartanair_train.py`) and the depth type (z-depth vs ray length) on the first trajectory.
5. **Both targets** from each model: the MI kernel (`colmap_information_kernel.py`, `sigma_pix` and priors fixed once for the whole dataset; TartanAir is metric) and the item cosine `|P_i ∩ P_j| / √(|P_i| |P_j|)` (PSD by construction). Eigenvalues logged per model — the MI kernel is not guaranteed PSD.

### Model

6. **MegaLoc frozen**, its 8448-d retrieval descriptor unchanged.
7. **The kernel descriptor comes from its own SALAD aggregator on the frozen DINOv2 patch tokens**, initialised as a copy of MegaLoc's SALAD (+ projection) and fine-tuned. One backbone pass serves both descriptors; the extra cost is a second aggregator.
8. **Plain copy first, structural changes as ablations.** Order of experiments:
    1. **Linear probe** — frozen MegaLoc, a linear map or small MLP `g` on the 8448-d output, `K̂_ij = ⟨ĝ(x_i), ĝ(x_j)⟩` with `ĝ = g/‖g‖`. Needs no GPU training beyond a head and no download of the full dataset; tells how much of the kernel is already in the descriptor and gives the baseline.
    2. **Plain SALAD branch.**
    3. **+ variable per-cluster mass** (learned or unbalanced Sinkhorn marginals instead of `κ = [1_m, n − m]`).
    4. **+ blocks weighted by evidence** (scale `V_k` by `√(mass_k / Σ mass)` before the final normalisation instead of pure intra-normalisation).

### Training

9. **Loss: masked, weighted regression of the batch kernel**, `L = Σ_{i≠j} m_ij · w_ij · (⟨x_i, x_j⟩ − K_ij)²`. `w_ij` balances pairs by bins of K so zeros do not dominate; `m_ij` masks unknown targets. Later ablations: Gaussian KL between `N(0, K_B)` and `N(0, X Xᵀ)` (penalises small-eigenvalue errors, but touchy on small batches and on an indefinite MI kernel), and a rank term if Spearman shows the ordering blurred.
10. **Sampler:** each sub-batch is 32–64 frames of one trajectory, stratified over K. Frames of different environments in the same step are free targets K = 0; pairs from different trajectories of the same environment are masked (unknown under decision 1).

### Validation

11. **Held-out environments**, related ones grouped: `office2` + `office` (indoor, 21 + 15 trajectories), `oldtown` (urban, 15), `seasonsforest` + `seasonsforest_winter` (nature, 16 + 19) — about 86 of 369. `abandonedfactory` + `abandonedfactory_night` stay together in training.
12. **Metrics:** kernel fit — Pearson and Spearman over off-diagonal pairs against both targets (`tools/compare_kernels.py`); the floor — mean K̂ over pairs with true K = 0; cull agreement — count-driven cull (`target_alive` = same N) on the learned and the GT kernel, Jaccard of the survivors. Retrieval needs no metric (unchanged by construction).
13. **Afterwards:** real sequences (REPLICA, SCANNETPLUSPLUS, ETH, TUM) with COLMAP kernels as a second test; end-to-end SLAM (ATE with information culling in AllFeature-VSLAM) on the final model only.

## Alternatives considered and rejected

| Decision | Rejected | Why |
|---|---|---|
| 1 per-trajectory models | per-environment models; per-trajectory now + environment later | needs an unverified shared world frame across TartanAir trajectories; much larger kernels and pair search. Cross-trajectory revisits (loop-closure-like positives) are therefore absent from the data |
| 2 stride 5 | overlap-based subsampling; all frames | overlap-based removes exactly the high-K pairs; all frames lets 30 Hz near-duplicates dominate at ~25× the cost |
| 3 GT frustum overlap | exhaustive; sequential; MegaLoc retrieval | O(n²); only temporal neighbours; MegaLoc's misses become false zeros in the target |
| 5 both targets | MI only; item cosine only | same expensive steps; the probe decides which one a global descriptor fits |
| 6–7 separate head, own SALAD | one shared descriptor; head on the 8448-d output; head on the 16640-d pre-projection; own last-4 blocks + SALAD | a shared descriptor changes retrieval; the 8448 head cannot reach the floor's source; the 16640 head only gets per-block weights fixed for every image; own blocks cost ≈ +30 % ViT and overfit the renderer |
| 8 plain copy first | both structural changes from the start; plain copy only | could not attribute a failure; would not test the two structural causes |
| 9 masked regression | Gaussian KL; regression + rank term | KL is touchy on small batches and indefinite targets; the rank term only if Spearman shows a need |
| 11 grouped held-out environments | leave-one-environment-out; random trajectories | ~15 runs per ablation; random trajectories measure scene memorisation |
| 12 fit + cull agreement | fit only; end-to-end ATE | a kernel can correlate well and still be wrong on the small eigenvalues the culler uses; ATE is noisy and confounded, final model only |

## Bring-up order

1. **Prerequisites:** commit `tools/colmap_gt_model.py` (untracked at `6bc3d30`, written for REPLICA); decide where the pipeline lives (open point 1). No local MegaLoc model clone exists (`tools/megaloc/` is created by `export_megaloc.py`), and TartanAir-train is not in the benchmark yet.
2. **One trajectory end to end:** download one environment group, stride 5, GT-overlap pairs, COLMAP features + custom-pairs matching, `colmap_gt_model.py` (its verification passes?), both kernels, eigenvalues, MI vs item-cosine comparison.
3. **Linear probe** (experiment 8.1) on a handful of training environments + one held-out group.
4. **Scale up** the data only if the probe or a first SALAD branch looks promising; then experiments 8.2–8.4.

## Open points

1. **Where the pieces live:** placecell `tools/` (next to the kernel tools), VSLAM-LAB (dataset access, COLMAP environment, sequence-target convention), or a separate training repo — the latter also keeps SALAD's GPL-3.0 training code out of placecell (Apache-2.0). MegaLoc's training code is not released; its `megaloc_model.py` (MIT) loads the weights.
2. **Which target to train on first** — the linear probe decides.
3. **Resources:** the full TartanAir-train download is a few hundred GB (≈ 10 GB per environment/difficulty group, 29 GB worst case, 36 groups); a GPU for steps 8.2–8.4. The probe runs on a few environments with descriptors from our TensorRT embedder.
4. **How descriptors are computed for training and the probe:** no batch-dump tool exists (`megaloc_embedder_smoke` is a smoke test). The SALAD branch needs the DINOv2 patch tokens, so training is PyTorch (MegaLoc via `torch.hub` / `megaloc_model.py`); the probe can use the same code, or a TensorRT dump if it must match the deployed embedder exactly.
5. **Export and runtime (only once a branch is worth deploying):** a second ONNX output in `tools/export_megaloc.py` with its name and size in the sidecar `.yaml`; `MegaLocEmbedder` returning two descriptors; the information kernel run with `centred = false`.

## References

- G. Berton, C. Masone. *MegaLoc: One Retrieval to Place Them All.* CVPRW 2025, arXiv:2502.17237; code `github.com/gmberton/MegaLoc` (MIT), weights HF `gberton/MegaLoc`.
- S. Izquierdo, J. Civera. *Optimal Transport Aggregation for Visual Place Recognition.* CVPR 2024, arXiv:2311.15937; code `github.com/serizba/salad` (GPL-3.0).
- X. Wang et al. *Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning.* CVPR 2019.
- W. Wang et al. *TartanAir: A Dataset to Push the Limits of Visual SLAM.* IROS 2020.
- A. Rau et al. *Predicting Visual Overlap of Images Through Interpretable Non-Metric Box Embeddings.* ECCV 2020 — asymmetric overlap; cited from memory, to verify.
