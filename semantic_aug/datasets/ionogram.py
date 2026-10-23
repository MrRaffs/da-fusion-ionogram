import os
import random
from glob import glob
from typing import Tuple, Any, Dict
from PIL import Image
import torch
import torchvision.transforms as transforms
from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation

DEFAULT_DATA_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "..", "..", "data", "ionogram_brin"
)

class IonogramDataset(FewShotDataset):
    num_classes: int = 4
    class_names = ["RangeSF", "FreqSF", "MixSF"]

    def __init__(self, 
                 *args, 
                 data_dir: str = DEFAULT_DATA_DIR,
                 split: str = "train", 
                 seed: int = 0,
                 examples_per_class: int = None,
                 generative_aug: GenerativeAugmentation = None,
                 synthetic_probability: float = 0.5,
                 image_size: Tuple[int, int] = (448, 448), 
                 **kwargs):

        super(IonogramDataset, self).__init__(
            *args, 
            examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            generative_aug=generative_aug, 
            **kwargs
        )

        self.data_dir = data_dir
        self.split = split
        self.seed = seed
        self.image_size = image_size

        # Transforms for standard evaluation/training
        train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.transform = train_transform if split == "train" else val_transform

        self.image_paths = []
        self.labels = []

        random.seed(seed)

        # Load image paths for binary classes
        for class_idx, class_name in enumerate(self.class_names):
            class_folder = os.path.join(self.data_dir, split, class_name)
            if not os.path.exists(class_folder):
                continue

            paths = []
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
                paths.extend(glob(os.path.join(class_folder, ext)))
            
            paths = sorted(list(set(paths)))
            random.shuffle(paths)

            # Apply few-shot filtering if specified
            if examples_per_class is not None and split == "train":
                paths = paths[:examples_per_class]

            self.image_paths.extend(paths)
            self.labels.extend([class_idx] * len(paths))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Apply generative augmentation on the fly during training if active
        if self.generative_aug is not None and self.split == "train":
            if random.random() < self.synthetic_probability:
                image = self.generative_aug(image_path, self.class_names[label])
                if isinstance(image, Image.Image):
                    return self.transform(image), label

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_image_by_idx(self, idx: int) -> Image.Image:
        """Returns raw PIL Image for Textual Inversion concept extraction."""
        return Image.open(self.image_paths[idx]).convert("RGB")

    def get_metadata_by_idx(self, idx: int) -> Dict[str, Any]:
        """Returns metadata dictionary required by fine_tune.py."""
        label = self.labels[idx]
        return {
            "name": self.class_names[label],
            "label": label,
            "image_path": self.image_paths[idx]
        }