import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from datetime import datetime
from .model_helper import *


def train(**kwargs) -> None:

    device = select_device() # Device selection (CPU or GPU)
    controlled_randomness() # Random seed for deterministic results (reproducible)
    check_model_name(kwargs["model_name"]) # check for correct model name

    # Load dataset
    image_size = kwargs["image_size"]
    size = (image_size, image_size) # (height, width) - Resize image to have a standard size
    train_dataset = get_training_data(size)
    val_dataset = get_validation_data(size)

    # Hyperparameters
    num_epochs = kwargs["num_epochs"]
    batch_size = kwargs["batch_size"]
    learning_rate = kwargs["learning_rate"]
    weight_decay = kwargs["weight_decay"]
    num_workers = kwargs["num_workers"]

    # Model parameters
    model_name = kwargs["model_name"]
    kwargs["num_classes"] = len(train_dataset.classes) # Number of classes in the dataset, ie: 4 -> {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}
    with_weights = kwargs["with_weights"]
    if with_weights:
        model_kwargs = get_model_params(model_name) # Get model parameters from last trained model
    else:
        model_kwargs = { 
            key: value 
                for key, value in kwargs.items()
                    if key in [ "in_channel", 
                                "num_classes",
                                "num_layers",
                                "num_heads",
                                "patch_size",
                                "image_size" ]
        }

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Set up logging directory
    log_dir = kwargs["log_dir"]
    log_dir = Path(log_dir) / Path(model_name) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Set up TensorBoard logging
    logger = SummaryWriter(log_dir)
    logger.add_text("model_name", model_name)
    if model_name in ["convblocks", "unet"]: # architecture graph in TensorBoard for convolutional models
        logger.add_graph(load_model(model_name, **model_kwargs), torch.zeros(batch_size, 3, *size)) # using images so 3 channels (RGB)
    elif model_name == "vit": # not traceable due to dynamic behavior, so use script instead
        scripted_model = torch.jit.script(load_model(model_name, **model_kwargs))
        logger.add_graph(scripted_model, torch.zeros(batch_size, 3, *size))
    images = torch.stack([train_loader.dataset[i][0] for i in range(100)]) # Get 100 images from the dataset
    image_grid = torchvision.utils.make_grid(images, nrow=10, normalize=True) # Create a grid of the 100 images
    logger.add_image("train_images", image_grid)

    # Model selection
    model = load_model(model_name, with_weights=with_weights, **model_kwargs)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # Training
    global_step = 0
    for epoch in range(num_epochs):

        model.train()
        train_accuracy = []

        # Training loop
        for data, label, in train_loader:
            data, label = data.to(device), label.to(device)
            
            # data = [batch_size, 3, height, width]
            # output = [batch_size, num_classes] --> for each batch --> [prob of class 0, prob of class 1, ...]
            # label = [batch_size] --> [label of batch 0, label of batch 1, ...]
            output = model(data)
            loss = loss_func(output, label)

            train_accuracy.extend((output.argmax(1) == label).cpu().detach().float().numpy())
            # check if the output equals the label,
            # then, detach from gpu (if so),
            # next, convert to float,
            # finally, convert to numpy array to extend the list for logging at the end of the epoch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        logger.add_scalar("train/accuracy", np.mean(train_accuracy), epoch) # (number of correct predictions) / (total number of predictions)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Accuracy: {np.mean(train_accuracy):.4f}, Train Loss: {loss.item():.4f}")
        

        # Validation loop
        model.eval()
        val_accuracy = []
        
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)

            with torch.inference_mode():
                output = model(data)
            val_accuracy.extend((output.argmax(1) == label).cpu().detach().float().numpy())
        
        logger.add_scalar("val/accuracy", np.mean(val_accuracy), epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {np.mean(val_accuracy):.4f}")


        logger.flush()

        print("----------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------")

    save_model(model) # Save the model weights to the model directory. (This location is used when loading model for future use)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th") # Save the model weights to the log directory
    print("Training completed.")
    print(f"Logs saved to {log_dir}. Model saved to model/{model_name}.th")
    print("----------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TensorBoard logging directory name
    parser.add_argument("--log_dir", type=str, default="logs", help="Name of directory to save logs.")

    # Hyperparameters
    parser.add_argument("--num_epochs", type=int, default=60, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="convblocks", required=True, help="Name of the model to train on. Options: convblocks, unet, vit.")
    parser.add_argument("--image_size", type=int, default=128, help="Resize the image to this size. Recommended: 128, 256, or 512. For faster training, use default 128.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the model. Only for Convblocks and ViT")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads for attention. Only for ViT.")
    parser.add_argument("--with_weights", type=bool, default=False, help="Load model weights from model directory, for further training.")

    # Static patameters
    parser.set_defaults(
        in_channel=3, # Static input channel of 3 for images (RGB)
        num_classes=2, # default to 2 for binary classification - dynamically sets based on the dataset
        patch_size=8, # based on image size, Only for ViT model
    )

    train_args = parser.parse_args()

    # Set patch size based on image size
    if train_args.image_size >= 512:
        train_args.patch_size = 16

    train(**vars(train_args))
