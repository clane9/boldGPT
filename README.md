# boldGPT 🧠

![boldGPT](.github/images/boldgpt.png)

Humans struggle to "see" the structure in functional MRI (BOLD) brain maps. Our goal is to train a GPT that understands brain maps better than humans. This kind of "foundation" model should be useful for things like phenotype prediction and brain activity decoding. Plus it will hopefully generate neat fake brain maps.

## Roadmap

- [x] Prepare the [natural scenes dataset](https://naturalscenesdataset.org/) for GPT consumption
  - [x] NSD "beta" activity vectors to flat maps ([NSD-Flat](https://github.com/clane9/NSD-Flat), [🤗](https://huggingface.co/datasets/clane9/NSD-Flat))
  - [x] Flat maps to tokenized patch sequences ([`BoldTokenizer`](boldgpt/tokenizer.py)) supporting raster, radial, and random order.
- [x] Implement model
  - [x] [`BoldGPT`](boldgpt/model.py) supporting causal/cross attention, masking, and shuffled prediction. Borrows elements from [timm](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py), [nanoGPT](https://github.com/karpathy/nanoGPT), and [BEiT](https://github.com/microsoft/unilm/blob/master/beit/modeling_pretrain.py).
- [x] Implement training script
  - [x] [`train_gpt.py`](scripts/train_gpt.py) for training models on shuffled next patch prediction.
- [ ] Train models
  - [x] [Initial training run](jobs/train_boldgpt_01) with ViT-small, shuffled next patch prediction with discrete token targets, 1000 epochs.
- [ ] Implement benchmarks
  - [ ] Masked patch prediction
  - [ ] Patch-wise object classification (i.e. segmentation)
  - [ ] [image-to-activity](https://codalab.lisn.upsaclay.fr/competitions/9304)
  - [ ] [activity-to-image](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)
- [ ] Evaluation
- [ ] Generate pretty brain maps

## Results

### Initial training run

![Train examples](.github/images/230914153355-numerous-horse_train_examples_89999.png)

In [our first training run](jobs/train_boldgpt_01), we trained a ViT-small decoder to auto-regressively predict the next patch (like GPT). However, the patch order is shuffled for each training example. So it's actually a bit like training a masked prediction (BERT/MAE) style model on all possible masking ratios simultaneously. The targets are discrete patch tokens from k-means (like HUBERT). We trained the model on [NSD-Flat](https://huggingface.co/datasets/clane9/NSD-Flat) for 1000 epochs (90K steps) with a batch size of 2048 on 4 V100 GPUs. The figure shows the patch order, activity data, target tokens, and predictions for a few training samples at the end of training.

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

If you'd like to contribute, please feel free fork the repo and join (or start!) a conversation in our [issues](https://github.com/clane9/boldGPT/issues).

## Citation

If you find this repository helpful, please consider citing:

```
@article{lane2023boldgpt,
  author       = {Connor Lane},
  title        = {boldGPT: A GPT foundation model for brain activity maps},
  howpublished = {\url{https://github.com/clane9/boldGPT}},
  year         = {2023},
}
```
