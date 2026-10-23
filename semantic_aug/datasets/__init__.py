from semantic_aug.datasets.pascal import PASCALDataset
from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.ionogram import IonogramDataset

DATASETS = {
    "pascal": PASCALDataset,
    "coco": COCODataset,
    "spurge": SpurgeDataset,
    "imagenet": ImageNetDataset,
    "ionogram": IonogramDataset,
}

__all__ = [
    "DATASETS",
    "IonogramDataset",
    "SpurgeDataset",
    "COCODataset",
    "PASCALDataset",
    "ImageNetDataset",
]