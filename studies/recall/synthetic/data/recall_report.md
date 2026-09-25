# MAJOR-only seeded-defect recall — results (final)

**detection@1** (single audit): 52/80 = **65%** (95% CI 54%-75%)
**detection@2** (union of 2 runs): 29/40 = **72%** (95% CI 57%-84%)

| category | detection@1 | detection@K |
|---|---|---|
| bug | 7/20 (35%) | 4/10 (40%) |
| difference | 14/20 (70%) | 8/10 (80%) |
| methodology | 16/20 (80%) | 9/10 (90%) |
| missing | 15/20 (75%) | 8/10 (80%) |

## Clean-verdict stress test
- paper 1023: 3/4 seeds found -> defects surfaced; baseline re-detected 0/4; emergent/run [0, 1]
- paper 913: 3/4 seeds found -> defects surfaced; baseline re-detected 0/9; emergent/run [2, 2]
- paper 2657: 2/4 seeds found -> defects surfaced; baseline re-detected 0/5; emergent/run [0, 0]
- paper 2578: 4/4 seeds found -> defects surfaced; baseline re-detected 0/5; emergent/run [3, 2]
- paper 2167: 4/4 seeds found -> defects surfaced; baseline re-detected 0/8; emergent/run [7, 3]
- paper 1717: 3/4 seeds found -> defects surfaced; baseline re-detected 0/12; emergent/run [0, 0]
- paper 1908: 3/4 seeds found -> defects surfaced; baseline re-detected 0/7; emergent/run [2, 0]
- paper 4090: 2/4 seeds found -> defects surfaced; baseline re-detected 0/6; emergent/run [2, 1]
- paper 1171: 3/4 seeds found -> defects surfaced; baseline re-detected 0/5; emergent/run [2, 0]
- paper 3463: 2/4 seeds found -> defects surfaced; baseline re-detected 0/8; emergent/run [0, 3]

## Misses (0-detection seeds -> qualitative autopsy targets)
- 1023/S07 [bug] Regression readout clamps all negative outputs
- 913/S07 [bug] Clamp final class logits before cross-entropy
- 2657/S03 [difference] Run a bottleneck autoencoder under the NormalizingFlow label
- 2657/S07 [bug] Bypass MCM's learned masks during reconstruction
- 1717/S06 [methodology] Fit the regression baselines jointly on training and test fMRI
- 1908/S07 [bug] Cancel Gaussian rotations when constructing covariance matrices
- 4090/S02 [missing] Remove the industrial-image contraction-flow model
- 4090/S07 [bug] Make every contraction unit non-invertible with ReLU
- 1171/S03 [difference] Use a plain twelve-layer convolutional backbone instead of ResNet-12
- 3463/S01 [missing] Remove all released AP-10K and PF-Pascal data-pipeline support
- 3463/S07 [bug] Clamp all extracted DiTF channels to nonnegative values

Leave-one-paper-out detection@1 range: [0.6111, 0.6389, 0.6389, 0.6389, 0.6389, 0.6528, 0.6528, 0.6667, 0.6667, 0.6944]
