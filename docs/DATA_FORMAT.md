# Input specification

The `.pkl` file must contain `train`, `valid`, and `test` dictionaries. Only load
pickle files from a trusted source; pickle can execute code during loading.

Each split has the following fields:

| Field | Type / shape | Meaning |
| --- | --- | --- |
| `id` | N strings | Original sample identifiers, unique within a split |
| `raw_text` | N strings | Raw utterances, not cached BERT embeddings |
| `regression_labels` | N or N x 1 | Signed humor targets or sentiment scores |
| `audio` | N x La x Da, float | Original unaligned acoustic features |
| `vision` | N x Lv x Dv, float | Original unaligned visual features |
| `audio_lengths` | N integers | Positive valid prefix lengths |
| `vision_lengths` | N integers | Positive valid prefix lengths |

| Dataset | Train / validation / test | Audio / vision dimensions | Targets |
| --- | --- | --- | --- |
| UR-FUNNY | 7614 / 980 / 994 | 81 / 371 | -1, +1 |
| CMU-MOSI | 1284 / 229 / 686 | 5 / 20 | [-3, 3] |
| CMU-MOSEI | 16326 / 1871 / 4659 | 74 / 35 | [-3, 3] |

UR-FUNNY uses the original 50-step zero-padded features. The sentiment datasets
retain the temporal lengths of their processed files; do not crop or align them
to the text tokens. Masks identify valid prefixes and exclude padding from
attention and pooling. The loader applies no audiovisual normalization,
resampling, modality dropout, split changes, or neutral-label removal.

The loader checks the split sizes and rejects overlapping IDs when multiple
splits are loaded. It records file and token hashes for the supplied data.
These checks cannot establish that a newly extracted feature file is identical
to the original features: retain the providers' preprocessing and split records.

For model-level use, `batch['text']` has shape `[B, 2, 64]`, containing token IDs
and the attention mask. `batch['audio']` and `batch['vision']` are float tensors;
their `*_mask` fields are Boolean `[B, L]` tensors. `target` is `[B]`.
