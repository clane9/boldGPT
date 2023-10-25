# boldGPT 🧠

![boldGPT](.github/images/boldgpt.png)

Humans struggle to "see" the structure in functional MRI (BOLD) brain maps. Our goal is to train a GPT that understands brain maps better than humans. This kind of "foundation" model should be useful for things like brain activity encoding and decoding. Plus it will hopefully generate neat fake brain maps.

## Overview

**Datasets.** We train our models using the [NSD-Flat](https://huggingface.co/datasets/clane9/NSD-Flat) Hugging Face dataset, which is derived from the [Natural Scenes Dataset](https://naturalscenesdataset.org/). This is a dataset of paired fMRI BOLD activation maps and natural images from COCO.

**Models.** All our models use a vanilla ViT architecture ([`Transformer`](boldgpt/models/transformer.py)) adapted from [timm](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py). We consider a few different pre-training objectives:

- Auto-regressive next patch prediction with shuffled patch order ([`ImageGPT`](boldgpt/models/gpt.py)). With either:

  - Discrete k-means token targets ([`KMeansTokenizer`](boldgpt/tokenizer.py)) and cross-entropy loss
  - Continuous patch targets and MSE loss

  The shuffling idea is somewhat new, although see also [SAIM](https://github.com/qiy20/SAIM/tree/main) and [RandSAC](https://arxiv.org/abs/2203.12054). Our main innovation compared to these works is the use of a [next position query embedding](boldgpt/models/transformer.py).

- Masked patch prediction following [MAE](https://github.com/facebookresearch/mae) ([`MAE`](boldgpt/models/mae.py))

**Evaluation.** We are primarily interested in two downstream tasks:

- Image-to-BOLD, i.e. fMRI encoding. See also the [Algonauts 2023 challenge](http://algonauts.csail.mit.edu/).
- BOLD-to-Image, i.e. fMRI image reconstruction. See also [MindEye](https://github.com/MedARC-AI/fMRI-reconstruction-NSD).

## Preliminary results

**Training examples for discrete token targets.**
![Train examples](.github/images/230914153355-numerous-horse_train_examples_89999.png)


**Training examples for continuous patch targets.**
![Train examples](.github/images/231024085643-stocky-coffeeshop_train_examples_48779.png)


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

## Contribute

This project is under active development in collaboration with [MedARC](https://www.medarc.ai/) and we welcome contributions or feedback! If you'd like to contribute, please feel free fork the repo and start a conversation in our [issues](https://github.com/clane9/boldGPT/issues), or join us on the MedARC discord server.

## Citation

If you find this repository helpful, please consider citing:

```
@misc{lane2023boldgpt,
  author       = {Connor Lane},
  title        = {boldGPT: A GPT foundation model for brain activity maps},
  howpublished = {\url{https://github.com/clane9/boldGPT}},
  year         = {2023},
}
```
