# MoIF
Official implementation of MoIF: Möbius Interaction Filtering with order-selective spectral regularization for multimodal humor detection.

**Paper:** *[Möbius Interaction Filtering: Order-Selective Spectral Regularization for Multimodal Humor Detection](https://arxiv.org/abs/2611.12913)*

[arXiv:2611.12913](https://arxiv.org/abs/2611.12913) | [PDF](https://arxiv.org/pdf/2611.12913) | [DOI](https://doi.org/10.48550/arXiv.2611.12913)

**Authors:** Jiahao Qin and Yiwen Wang

MoIF learns seven modality-subset predictions and implements an order-selective
filter through their uniform aggregate. Text uses fine-tuned RoBERTa-large;
private temporal encoders and balanced reciprocal attention construct the
unimodal, bimodal, and trimodal states.

This initial release contains **MoIF code only**: model, input pipeline,
training, evaluation, a Mobius decomposition utility, configurations, and tests.
Datasets, pretrained language-model weights, trained checkpoints, baseline
implementations, manuscript files, and experiment outputs are not included.

## Installation

Use Python 3.10 or later and install a PyTorch build appropriate for your GPU.
The original runs used PyTorch 2.0.1/CUDA 11.8 on V100 and PyTorch 2.8.0/CUDA
12.8 on RTX 5090, with Transformers 4.35.0 and NumPy 1.26.4. A 5090 requires a
compatible recent PyTorch/CUDA build; the V100 environment is not interchangeable.

```bash
git clone https://github.com/D-ST-Sword/MoIF.git
cd MoIF
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Tests use small synthetic encoders and inputs; they do not download models or
count as benchmark experiments. Run commands from the repository root.

## Data

Obtain the datasets from their providers and respect their access terms:

- [UR-FUNNY](https://github.com/ROC-HCI/UR-FUNNY)
- [CMU-MOSI and CMU-MOSEI / CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)

The loaders consume the **processed, unaligned feature pickle**, not raw media.
See [the input specification](docs/DATA_FORMAT.md) for required fields, feature
dimensions, masks, and splits. The repository does not redistribute or regenerate
the original audiovisual features. Different feature extractions are not the
paper's input protocol.

`--pretrained` accepts `roberta-large` or a local Hugging Face snapshot containing
its weights and tokenizer. The text encoder and tokenizer must use the same
snapshot. Tokenization uses raw utterances with a maximum length of 64.

## Training

```bash
python -m moif.train --dataset urfunny --data /path/to/urfunny_unaligned_50.pkl --seed 42 --output runs/urfunny_seed42
```

Use `--dataset mosi` or `--dataset mosei` with the corresponding processed file.
Each run uses AdamW, batch size 32, learning rate 1e-5, weight decay 1e-4,
cosine decay to 1e-7 over 40 epochs, and gradient clipping at 1.0. The objective
is L1 loss on the aggregate plus 0.1 times mean subset L1 loss. CUDA autocast
is limited to the text backbone; fusion stays in float32.

The checkpoint with minimum validation MAE is selected. Early stopping requires
at least 20 epochs and 20 stale epochs. Test evaluation occurs after selection.
Existing output directories are rejected, preventing accidental run replacement.
`--smoke` performs two training batches and validation only; it is not a result seed.

| Dataset | Five training seeds |
| --- | --- |
| UR-FUNNY | 7, 22, 42, 93, 99 |
| CMU-MOSI | 22, 42, 93, 123, 456 |
| CMU-MOSEI | 22, 42, 93, 123, 456 |

The settings are also listed in [configs/paper.json](configs/paper.json).
Training uses the fixed settings in `moif/train.py`; the JSON is a configuration
record, not an extra override layer.

## Evaluation

```bash
python -m moif.evaluate --dataset urfunny --data /path/to/urfunny_unaligned_50.pkl --checkpoint runs/urfunny_seed42/selected.pt --output runs/urfunny_seed42_replay
```

Outputs include `metrics.json` and per-example `predictions.npz` with all eligible
subset scores. Use `--split train` or `--split valid` for the other splits.
`--available audio vision` executes only A, V, and AV, without calling the text
encoder. Complete-input evaluation is the default.

Classification metrics are stored as fractions, not percentages. UR-FUNNY uses
signed labels and a threshold of `>= 0`. For MOSI/MOSEI, the paper's binary metric
is `has0_ge_zero`: neutral samples are retained on the nonnegative side. The
evaluator additionally reports `has0_gt_zero`, `non0_gt_zero`, and Acc-3/5/7 so
that evaluation conventions remain explicit.

The released model preserves the original `core.*` checkpoint parameter names.
Legacy checkpoints may store `variant="fixed_gate"`; this is the same final
MoIF computation. Each attention pair retains frozen, unused `gate` tensors to
preserve checkpoint keys and initialization RNG. They are never evaluated and
are not a learned routing component. Legacy checkpoints with pickled metadata
may require `--trusted-checkpoint`; use this only with files you trust.

## Interaction coordinates

```bash
python -m moif.spectral --predictions runs/urfunny_seed42_replay/predictions.npz --output runs/urfunny_seed42_spectrum.npz
```

For `f(empty)=0`, the Mobius increments are inclusion-exclusion contrasts of the
seven subset predictions. Their order sums `g1, g2, g3` satisfy:

```text
f(TAV) = g1 + g2 + g3
MoIF   = (4*g1 + 2*g2 + g3) / 7 = mean(f(T), ..., f(TAV))
```

This utility exposes an equivalent coordinate view, not an additional network,
loss, calibration step, or change to the inference rule.

## Implementation provenance

The release packages the computation used by the reported model, removing
machine-specific paths and unrelated experiment infrastructure. See
[source provenance](docs/source_provenance.json) for source-object hashes.
The public interface supports user-selected data and pretrained-model locations;
network dimensions, directional weighting, loss, and evaluation definitions are
unchanged. Third-party libraries remain subject to their respective licenses.

## Citation

If you use MoIF in your research, please cite our paper:

```bibtex
@misc{qin2026moif,
  title         = {{M{\"o}bius Interaction Filtering: Order-Selective Spectral Regularization for Multimodal Humor Detection}},
  author        = {Jiahao Qin and Yiwen Wang},
  year          = {2026},
  eprint        = {2611.12913},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MM},
  doi           = {10.48550/arXiv.2611.12913},
  url           = {https://arxiv.org/abs/2611.12913}
}
```

The repository also provides [CITATION.cff](CITATION.cff) for GitHub's
**Cite this repository** feature.
