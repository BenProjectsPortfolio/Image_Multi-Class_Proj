import torch
from pathlib import Path


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU()
        )
    
    def forward(self, x):
        return self.double_conv(x)


class ConvolutionalUNet(torch.nn.Module):  
    class DownConv(torch.nn.Module):
        def __init__(self, in_channel, out_channel):
            super().__init__()
            self.double_conv = DoubleConv(in_channel, out_channel)
            self.maxpool = torch.nn.MaxPool2d(2)
        def forward(self, x):
            x = self.double_conv(x)
            return x, self.maxpool(x)
        
    class UpConv(torch.nn.Module):
        def __init__(self, in_channel, out_channel):
            super().__init__()
            self.up_conv = torch.nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
            self.double_conv = DoubleConv(in_channel, out_channel)
        def forward(self, x, skip):
            x = self.up_conv(x)
            x = torch.cat((x, skip), dim=1) # skip connection (residual connection)
            x = self.double_conv(x)
            return x
    
    def __init__(self, in_channel = 3, num_classes = 2, num_layers = None, num_heads = None, patch_size = None, image_size = None):
        super().__init__()
        self.down1 = self.DownConv(in_channel, 16)
        self.down2 = self.DownConv(16, 32)
        self.bridge = DoubleConv(32, 64)
        self.up1 = self.UpConv(64, 32)
        self.up2 = self.UpConv(32, 16)
        self.classifier = torch.nn.Conv2d(16, num_classes, kernel_size=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1, x_max1 = self.down1(x)
        x2, x_max2 = self.down2(x_max1)
        x = self.bridge(x_max2)
        x = self.up1(x, x2)
        x = self.up2(x, x1)
        x = self.classifier(x)
        x = self.pool(x)
        x = torch.nn.Flatten()(x)
        return x


class ConvolutionalBlockNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channel, out_channel, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2
            self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
            self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size, 1, padding)
            self.conv3 = torch.nn.Conv2d(out_channel, out_channel, kernel_size, 1, padding)
            self.relu1 = torch.nn.ReLU()
            self.relu2 = torch.nn.ReLU()
            self.relu3 = torch.nn.ReLU()
        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.relu3(self.conv3(x))
            return x
        
    def __init__(self, in_channel = 3, num_classes = 2, num_layers = 3, num_heads = None, patch_size = None, image_size = None):
        super().__init__()
        layers = [
            torch.nn.Conv2d(in_channel, 24, kernel_size=11, stride=2, padding=5),
            torch.nn.ReLU(),
        ]
        current_channel = 24
        for _ in range(num_layers):
            next_channel = current_channel * 2
            layers.append(self.Block(current_channel, next_channel, stride=2))
            current_channel = next_channel
        layers.append(torch.nn.Conv2d(current_channel, num_classes, kernel_size=1))
        layers.append(torch.nn.AdaptiveAvgPool2d(1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        x = torch.nn.Flatten()(x)
        return x


class TransformerNet(torch.nn.Module):
    class PositionalEmbedding(torch.nn.Module):
        def __init__(self, num_patches, embed_dim):
            super().__init__()
            self.positional_embedding = torch.nn.Parameter(torch.randn(1, num_patches, embed_dim)) # learnable positional embedding
        def forward(self, x):
            return x + self.positional_embedding # Add positional embedding to input
    class TransformerLayer(torch.nn.Module):
        def __init__(self, embed_dim, num_heads):
            super().__init__()
            self.attention = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim * 4),
                torch.nn.ReLU(),
                torch.nn.Linear(embed_dim * 4, embed_dim),
            )
            self.norm1 = torch.nn.LayerNorm(embed_dim)
            self.norm2 = torch.nn.LayerNorm(embed_dim)
        def forward(self, x):
            x_norm = self.norm1(x)
            x = x + self.attention(x_norm, x_norm, x_norm)[0] # [0] is the attention outputs, [1] is the attention weights
            x = x + self.mlp(self.norm2(x))
            return x
    def __init__(self, in_channel = 3, num_classes = 2, num_layers = 3, num_heads = 4, patch_size = 8, image_size = 128):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        patched_in_channel = in_channel * patch_size * patch_size
        self.positional_encoding = self.PositionalEmbedding(num_patches, patched_in_channel)
        self.network = torch.nn.Sequential(
            *[
                self.TransformerLayer(embed_dim=patched_in_channel, num_heads=num_heads) for _ in range(num_layers)
            ],
            torch.nn.Linear(patched_in_channel, num_classes), # classifier
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = torch.nn.Flatten()

    def patch_image(self, image: torch.Tensor, patch_size: int) -> torch.Tensor:
        batch_size, channels, height, width = image.shape 
        unfolded = self.unfold(image).to(image.device)
        # after unfold:
        # [ batch_size, (channels * patch_size * patch_size), ((height // patch_size) * (width // patch_size))) ]
        num_patches = (height // patch_size) * (width // patch_size)
        # switch last two dimensions and ensure shape is correct
        patched_image = unfolded.permute(0, 2, 1).reshape(batch_size, num_patches, channels * patch_size * patch_size)
        return patched_image
    
    def forward(self, x):
        x = self.patch_image(x, self.patch_size)
        x = self.positional_encoding(x)
        x = self.network(x)
        x = x.permute(0, 2, 1) # switch last two dims: [32, 256, 4] -> [32, 4, 256]
        x = x.unsqueeze(-1) # add singleton dimension to perform adaptive avg pooling: [32, 4, 256] -> [32, 4, 256, 1]
        x = self.pool(x) # [32, 4, 256, 1] -> [32, 4, 1, 1]
        x = self.flatten(x) # [32, 4, 1, 1] -> [32, 4]
        return x
