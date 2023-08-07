# NSD-Flat

A [Huggingface dataset](https://huggingface.co/datasets/clane9/NSD-Flat) of pre-processed brain activity flat maps from the [Natural Scenes Dataset](https://naturalscenesdataset.org/), constrained to a visual cortex region of interest and rendered as PNG images.

## Building the dataset

### 1. Download source data

Run [`download_data.sh`](download_data.sh) to download the required source data:

- NSD stimuli images and presentation info
- COCO annotations
- NSD beta activity maps in fsaverge surface space

```bash
bash download_data.sh
```

### 2. Convert the COCO annotations

Run  [`convert_nsd_annotations.py`](convert_nsd_annotations.py) to crop and reorganize the COCO annotations for NSD.

```bash
python convert_nsd_annotations.py
```

### 3. Generate the dataset

Run [`generate_dataset.py`](generate_dataset.py) to generate the huggingface dataset in Arrow format.

```bash
python generate_dataset.py --img_size 256 --workers 8
```

## Load the dataset

Load the dataset from a local directory

```python
from datasets import load_from_disk

dataset = load_from_disk("processed/size-256")
```

Or load from [Huggingface Hub](https://huggingface.co/datasets/clane9/NSD-Flat)

```python
from datasets import load_dataset

dataset = load_dataset("clane9/NSD-Flat")
```
