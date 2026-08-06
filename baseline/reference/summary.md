# Frozen baseline reference result

Method: `hierarchical max(B0, composition) + S0 + S1`

Protocol: MVTec LOCO, five-category macro image-level AUROC, full-data,
training-free, four K-means seeds (0, 1, 2, 3).

| Object | Logical | Structural | Overall |
|---|---:|---:|---:|
| breakfast_box | 0.9853 | 0.8756 | 0.9282 |
| juice_bottle | 0.9529 | 0.9622 | 0.9566 |
| pushpins | 0.9578 | 0.8604 | 0.9119 |
| screw_bag | 0.9261 | 0.9171 | 0.9227 |
| splicing_connectors | 0.8806 | 0.8883 | 0.8840 |
| **Macro** | **0.9405** | **0.9007** | **0.9207** |

The executable reproduction writes the full-precision JSON and per-image CSV
under the configured `output_root`.
