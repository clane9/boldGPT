"""
Generate a JSON-lines file containing all NSD dataset detection and caption annotations,
converted from their original COCO sources.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from skimage.transform import resize
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

StrOrPath = Union[str, Path]
Bbox = Tuple[float, float, float, float]

NSD_DIR = "data/NSD"
COCO_ANNO_DIR = "data/COCO/annotations"


def load_nsd_annotations(
    stim_info_path: StrOrPath, coco_annotations_dir: StrOrPath
) -> List[Dict[str, Any]]:
    """
    Load and convert NSD image annotations from their COCO sources.
    """
    # All images in NSD are scaled to 425 x 425
    scaled_height, scaled_width = 425, 425
    stim_info = pd.read_csv(stim_info_path, index_col=0)

    cocos: Dict[Tuple[str, str], COCO] = {}
    for typ in ["instances", "captions"]:
        for split in ["train2017", "val2017"]:
            path = Path(coco_annotations_dir) / f"{typ}_{split}.json"
            cocos[(typ, split)] = COCO(path)

    def load_img_anns(coco: COCO, imgId: int):
        return coco.loadAnns(coco.getAnnIds(imgId))

    annotations = []
    for ii in tqdm(range(len(stim_info))):
        row = stim_info.iloc[ii].to_dict()
        nsd_id: int = row["nsdId"]
        assert nsd_id == ii

        split: str = row["cocoSplit"]
        coco_id: int = row["cocoId"]
        crop: Bbox = ast.literal_eval(row["cropBox"])

        img = cocos[("instances", split)].loadImgs(coco_id)[0]
        height, width = img["height"], img["width"]

        instances = load_img_anns(cocos[("instances", split)], coco_id)
        instances = [
            _crop_and_scale_instance(
                instance, crop, height, width, scaled_height, scaled_width
            )
            for instance in instances
        ]
        objects = _filter_and_reshape_instances(
            instances, cocos[("instances", split)].cats
        )
        captions = load_img_anns(cocos[("captions", split)], coco_id)
        captions = [obj["caption"] for obj in captions]

        record = {
            "nsd_id": nsd_id,
            "coco_split": split,
            "coco_id": coco_id,
            "height": scaled_height,
            "width": scaled_width,
            "objects": objects,
            "captions": captions,
        }
        annotations.append(record)

    return annotations


def _crop_shape(crop: Bbox, height: int, width: int) -> Tuple[int, int]:
    """
    Find the new image shape after cropping
    """
    cx1, cy1, cx2, cy2 = _crop2xyxy(crop, height, width)
    crop_height = round(cy2 - cy1)
    crop_width = round(cx2 - cx1)
    assert crop_height == crop_width
    return crop_height, crop_width


def _crop_and_scale_instance(
    instance: Dict[str, Any],
    crop: Bbox,
    height: int,
    width: int,
    scaled_height: int,
    scaled_width: int,
):
    """
    Crop a COCO instance.

    Args:
        instance: COCO detection/segmentation instance
        crop: a tuple of four numbers representing the amount cropped from each side of
            the image. The format is (top, bottom, left, right) in fractions of image
            size.
        height: original image height
        width: original image width
        scaled_height: scaled image height
        scaled_width: scaled image width

    References:
        https://cvnlab.slite.page/p/NKalgWd__F/Experiments#:~:text=it%20is%20useful.-,Column%204%20(cropBox),-is%20a%20tuple
    """
    bbox = instance["bbox"]
    bbox = _xywh2xyxy(bbox)
    bbox = _crop_and_scale_points(
        bbox, crop, height, width, scaled_height, scaled_width
    )
    bbox = list(_xyxy2xywh(bbox))
    assert 0 <= bbox[2] <= scaled_width
    assert 0 <= bbox[3] <= scaled_height

    segmentation = instance["segmentation"]
    if isinstance(segmentation, dict):
        segmentation, area = _crop_and_scale_rle(
            segmentation, crop, height, width, scaled_height, scaled_width
        )
    else:
        segmentation = [
            _crop_and_scale_points(
                seg, crop, height, width, scaled_height, scaled_width
            )
            for seg in segmentation
        ]
        segmentation = [seg.tolist() for seg in segmentation]
        area = _poly_area(segmentation, scaled_height, scaled_width)

        # Wrap poly segmentation in a dict so we at least have a consistent type. This
        # breaks from coco but we are not really adhering tightly to coco anyway.
        # NOTE: dealing with segmentations was too much work.
        segmentation = {"poly": segmentation}
    assert 0 <= area <= scaled_height * scaled_width

    cropped_instance = instance.copy()
    cropped_instance.update(bbox=bbox, segmentation=segmentation, area=area)
    return cropped_instance


def _crop_and_scale_points(
    points: np.ndarray,
    crop: Bbox,
    height: int,
    width: int,
    scaled_height: int,
    scaled_width: int,
) -> np.ndarray:
    """
    Crop and scale an array of points in original image pixel coordinates.
    """
    points = np.asarray(points)
    assert points.ndim == 1 or (
        points.ndim == 2 and points.shape[1] == 2
    ), "Invalid points"
    is_flat = points.ndim == 1
    points = points.reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]

    # clip points to the crop bounding box and reset to the crop origin
    cx1, cy1, cx2, cy2 = _crop2xyxy(crop, height, width)
    x = np.clip(x, cx1, cx2) - cx1
    y = np.clip(y, cy1, cy2) - cy1
    points = np.stack([x, y], axis=1)

    # scale and round
    crop_height, crop_width = _crop_shape(crop, height, width)
    points = points * [scaled_width / crop_width, scaled_height / crop_height]
    points = np.round(points, 2)
    assert np.all((0 <= points[:, 0]) & (points[:, 0] <= scaled_width))
    assert np.all((0 <= points[:, 1]) & (points[:, 1] <= scaled_height))

    if is_flat:
        points = points.flatten()
    return points


def _crop_and_scale_rle(
    segmentation: Dict[str, Any],
    crop: Bbox,
    height: int,
    width: int,
    scaled_height: int,
    scaled_width: int,
):
    mask = mask_util.decode(mask_util.frPyObjects(segmentation, height, width))
    cx1, cy1, cx2, cy2 = np.round(_crop2xyxy(crop, height, width)).astype(int)

    mask = mask[cy1:cy2, :][:, cx1:cx2]
    mask = resize(mask, (scaled_height, scaled_width), order=0)
    mask = np.asfortranarray(mask)
    area = int(mask.sum())
    segmentation = mask_util.encode(mask)
    segmentation["counts"] = segmentation["counts"].decode("utf-8")
    return segmentation, area


def _poly_area(points: List[List[float]], height: int, width: int) -> int:
    rle = mask_util.frPyObjects(points, height, width)
    area = int(mask_util.area(rle)[0])
    return area


def _crop2xyxy(crop: Bbox, height: int, width: int) -> Bbox:
    """
    Convert an NSD-style crop to a bounding box in xyxy coordinates.
    """
    # convert crop to xyxy format in pixel units
    top, bottom, left, right = crop
    cx1 = left * width
    cy1 = top * height
    cx2 = (1 - right) * width
    cy2 = (1 - bottom) * height
    return cx1, cy1, cx2, cy2


def _xywh2xyxy(bbox: Bbox) -> Bbox:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def _xyxy2xywh(bbox: Bbox) -> Bbox:
    x1, y1, x2, y2 = bbox
    return x1, y1, x2 - x1, y2 - y1


def _filter_and_reshape_instances(
    instances: List[Dict[str, Any]],
    cats: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Reshape instances from a list of record to a dict of columns orientation.
    """
    objects = {
        "target": [],
        "category": [],
        "supercategory": [],
        "bbox": [],
        "segmentation": [],
        "area": [],
        "iscrowd": [],
    }

    for obj in instances:
        area = obj["area"]
        assert area >= 0
        if area == 0:
            continue
        objects["target"].append(obj["category_id"])
        cat = cats[obj["category_id"]]
        objects["category"].append(cat["name"])
        objects["supercategory"].append(cat["supercategory"])
        objects["bbox"].append(obj["bbox"])
        objects["segmentation"].append(obj["segmentation"])
        objects["area"].append(area)
        objects["iscrowd"].append(obj["iscrowd"])
    return objects


if __name__ == "__main__":
    stim_info_path = Path(NSD_DIR) / "nsddata/experiments/nsd/nsd_stim_info_merged.csv"
    annotations = load_nsd_annotations(stim_info_path, COCO_ANNO_DIR)

    with open("data/nsd_annotations.jsonl", "w") as f:
        for annot in annotations:
            print(json.dumps(annot), file=f)
