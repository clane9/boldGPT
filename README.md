# boldGPT 🧠

![boldGPT: a GPT that understands brain maps better than humans](.github/images/boldgpt.png)

Humans struggle to "see" the structure in functional MRI (BOLD) brain maps. Our goal is to train a GPT that understands brain maps better than humans. This kind of "foundation" model should be useful for things like phenotype prediction and brain activity decoding. Plus it will probably generate neat fake brain maps.

## Installation

Clone the repo and run `poetry install`.

## Roadmap

[ ] Pre-process the natural scenes dataset for GPT consumption
  [ ] Extract visual cortex ROI activity
  [ ] Flat map projection
  [ ] Parcellate into square patches ("patchify")
  [ ] Encode patch activities as discrete integer "tokens" ("tokenize")
[ ] Implement baselines
  [ ] PCA masked patch prediction
[ ] Adapt model and train code from popular GPT libraries
  [ ] [nanoGpt](https://github.com/karpathy/nanoGPT)
  [ ] [facebookresearch/mae](https://github.com/facebookresearch/mae) (not actually GPT but close enough)
[ ] Figure out what to do about patch order
  [ ] Fixed arbitrary order
  [ ] Fixed hand-designed order
  [ ] Shuffled order
  [ ] Ignore order
[ ] Train models
  [ ] Small models on colab
  [ ] Bigger models on HPC
[ ] Evaluate masked prediction
[ ] Generate pretty brain maps


## Contribute

- Open an issue.
- Work on an issue.
