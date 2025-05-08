import argparse
import torch
from torch.utils.data import DataLoader
from .model_helper import *


def test(
        model: torch.nn.Module,
        dataset: DataLoader,
        ) -> None:
    
    print("Testing model")
    print("----------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------")
    print("Testing..", end="")

    model.eval()
    test_accuracy = []

    for data, label in dataset:
        data, label = data.to(device), label.to(device)

        with torch.inference_mode():
            output = model(data)

        test_accuracy.extend((output.argmax(1) == label).cpu().detach().float().numpy()) # if output equals label
        print(".", end="", flush=True) # progress bar, each interation is a dot

    print(f"\nTest Accuracy: {np.mean(test_accuracy):.4f}") # (number of correct predictions) / (total number of predictions)
    print("----------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="convblocks", required=True, help="Name of the model to test on. Options: convblocks, unet, vit.")
    parser.add_argument("--image_size", type=int, default=128, help="Resize the image to this size. Recommended sizes: 128, 256, or 512. Default: 128.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser_args = parser.parse_args()

    device = select_device() # Select the device to use (CPU or GPU)
    
    # Load the model
    model_name = parser_args.model_name
    check_model_name(model_name) # if one of the three options: convblocks, unet, vit
    with_weights = True
    model_kwargs = get_model_params(model_name) # loads given model's parameters during training
    model = load_model(model_name, with_weights=with_weights, **model_kwargs)
    model = model.to(device)

    # Load the test dataset
    image_size = parser_args.image_size
    size = (image_size, image_size) # (height, width)
    test_dataset = get_test_data(size)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=parser_args.batch_size,
        shuffle=False,
        num_workers=parser_args.num_workers,
    )

    test(model, test_loader) # Test the model

    print("Testing completed.")
    print("----------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------")
