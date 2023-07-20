# boldGPT 🧠

![boldGPT: a GPT that understands brain maps better than humans](.github/images/boldgpt.png)

Humans struggle to "see" the structure in functional MRI (BOLD) brain maps. Our goal is to train a GPT that understands brain maps better than humans. This kind of "foundation" model should be useful for things like phenotype prediction and brain activity decoding. Plus it will hopefully generate neat fake brain maps.

## Installation

Clone the repo and install the package in a new virtual environment

```bash
git clone https://github.com/clane9/boldGPT.git
cd boldGPT

python3 -m venv --prompt boldgpt .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Roadmap

- [ ] Pre-process the [natural scenes dataset](https://naturalscenesdataset.org/) for GPT consumption
  - [ ] NSD "beta" activity vectors to flat maps
  - [ ] Flat maps to tokenized patch sequences
- [ ] Implement baselines
  - [ ] PCA masked patch prediction
- [ ] Adapt model and train code from popular GPT libraries
  - [ ] [nanoGpt](https://github.com/karpathy/nanoGPT)
  - [ ] [facebookresearch/mae](https://github.com/facebookresearch/mae) (not actually GPT but close enough)
- [ ] Figure out what to do about patch order
  - [ ] Fixed arbitrary order
  - [ ] Fixed hand-designed order
  - [ ] Shuffled order
  - [ ] Ignore order (i.e. non-auto-regressive MAE)
- [ ] Train models
  - [ ] Small models on colab
  - [ ] Bigger models on HPC
- [ ] Evaluate masked patch prediction
- [ ] Generate pretty brain maps


## Contribute

If you'd like to contribute, please feel free fork the repo and join (or start!) a conversation in our [issues](https://github.com/clane9/boldGPT/issues).
