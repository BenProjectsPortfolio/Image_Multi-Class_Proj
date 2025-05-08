import torch
import json
import torchvision
import numpy as np
from torchvision.datasets import ImageFolder
from pathlib import Path
from .models import ConvolutionalUNet, TransformerNet, ConvolutionalBlockNet


PARENT_DIR = Path(__file__).resolve().parent
SEED = 2025
MODEL = {
    "convblocks": ConvolutionalBlockNet,
    "unet": ConvolutionalUNet,
    "vit": TransformerNet,
}


def load_model(
        model_name: str, 
        with_weights: bool = False, 
        **model_kwargs,
        ) -> torch.nn.Module:

    model = MODEL[model_name](**model_kwargs)

    with open(f"{PARENT_DIR}/{model_name}_configs.txt", "w") as f:
        json.dump(model_kwargs, f, indent=4)

    if with_weights:
        model_path = PARENT_DIR / f"{model_name}.th"
        assert model_path.exists(), f"Model weights not found at {model_path}"

        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError:
            raise AssertionError(
                f"Failed to load {model_path.name}."
            )
    
    return model


def save_model(
        model: torch.nn.Module,
        ) -> str:
    
    model_name = None

    for name, model_class in MODEL.items():
        if isinstance(model, model_class):
            model_name = name
            break
    
    if model_name is None:
        raise ValueError("Not supported model type.")
    
    output_path = PARENT_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def get_model_params(model_name: str) -> dict[str, int]:
    with open (f"{PARENT_DIR}/{model_name}_configs.txt", "r") as f:
            return json.load(f)


def select_device() -> torch.device:

    if torch.cuda.is_available(): # NVIDIA GPU
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built(): # Apple Silicon GPU
        device = torch.device("mps")
    else:
        print("No suitable device found, using CPU.")
        device = torch.device("cpu")

    return device


def controlled_randomness() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def check_model_name(model_name: str) -> None:
    if model_name not in ["convblocks", "unet", "vit"]:
        raise ValueError(f"Invalid model name: {model_name}. Must be one of the following options: convblocks, unet, vit.")


def get_training_data(
        size: tuple[int, int], # (height, width)
        ) -> ImageFolder:
    
    # Image Transformation format
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=size, antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        torchvision.transforms.RandomHorizontalFlip(p=0.5), # data augmentation for training data
        torchvision.transforms.RandomVerticalFlip(p=0.5), # 50% chance to flip image
    ])

    # Load dataset
    train_dataset = ImageFolder(root="data/training", transform=train_transform)
    # ImageFolder expects the directory structure to be: data/training/class_name/image.jpg
    # dataset[i] = (image_tensor, label_index) and dataset.classes = the label names
    return train_dataset


def get_validation_data(
        size: tuple[int, int], # (height, width)
        ) -> ImageFolder:

    # Image Transformation format
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    val_dataset = ImageFolder(root="data/validation", transform=val_transform)
    # ImageFolder expects the directory structure to be: data/training/class_name/image.jpg
    # dataset[i] = (image_tensor, label_index) and dataset.classes = the label names
    return val_dataset


def get_test_data(
        size: tuple[int, int], # (height, width)
        ) -> ImageFolder:

    # Image Transformation format
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    test_dataset = ImageFolder(root="data/testing", transform=test_transform)
    # ImageFolder expects the directory structure to be: data/training/class_name/image.jpg
    # dataset[i] = (image_tensor, label_index) and dataset.classes = the label names
    return test_dataset
