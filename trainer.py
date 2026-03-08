import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from safetensors.torch import save_file

def plot_multiple_images(images, n_cols, epoch):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    # Convert from CHW to HWC format for plotting
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        image = ((image + 1) / 2) # scale back
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")
    plt.savefig(f'{args.images_output_path}epoch_{epoch}.png')
    plt.close()  # Close the figure to free memory

class ImageDataset(Dataset):
    def __init__(self, file_paths, image_size, image_channels):
        self.file_paths = file_paths
        self.image_size = image_size
        self.image_channels = image_channels
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * image_channels, [0.5] * image_channels)  # Scale to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGBA' if self.image_channels == 4 else 'RGB')
        image = self.transform(image)
        return image

def get_dataloader(inputs, batch_size, image_size, image_channels):
    if type(inputs) == dict:
        file_paths = inputs["paths"].tolist()
    else:
        file_paths = glob.glob(f"{inputs}/*")
    
    dataset = ImageDataset(file_paths, image_size, image_channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    return dataloader

def discriminator_loss(real_output, fake_output, criterion):
    real_loss = criterion(real_output, torch.ones_like(real_output))
    fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, criterion):
    return criterion(fake_output, torch.ones_like(fake_output))

def train_step(images, batch_size, codings_size, generator, discriminator, gen_optimizer, disc_optimizer, criterion, device):
    noise = torch.randn(batch_size, codings_size, device=device)
    
    # Train Discriminator
    disc_optimizer.zero_grad()
    generated_images = generator(noise)
    real_output = discriminator(images)
    fake_output = discriminator(generated_images.detach())
    disc_loss = discriminator_loss(real_output, fake_output, criterion)
    disc_loss.backward()
    disc_optimizer.step()
    
    # Train Generator
    gen_optimizer.zero_grad()
    fake_output = discriminator(generated_images)
    gen_loss = generator_loss(fake_output, criterion)
    gen_loss.backward()
    gen_optimizer.step()
    
    return gen_loss.item(), disc_loss.item()

def train(dataloader, epochs, batch_size, codings_size, generator, discriminator, gen_optimizer, disc_optimizer, criterion, device):
    generator.train()
    discriminator.train()
    
    for epoch in range(epochs):
        for image_batch in dataloader:
            image_batch = image_batch.to(device)
            gen_loss, disc_loss = train_step(image_batch, batch_size, codings_size, generator, discriminator, 
                                             gen_optimizer, disc_optimizer, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs} - Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
        if args.images_output_path:
            generator.eval()
            with torch.no_grad():
                noise = torch.randn(batch_size, codings_size, device=device)
                display_images = generator(noise)
                plot_multiple_images(display_images, 8, epoch)
            generator.train()

class Generator(nn.Module):
    def __init__(self, codings_size, image_size, image_channels):
        super(Generator, self).__init__()
        
        self.fc = nn.Linear(codings_size, 6 * 6 * 256, bias=False)
        self.bn1 = nn.BatchNorm1d(6 * 6 * 256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.conv_transpose1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv_transpose2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv_transpose3 = nn.ConvTranspose2d(64, image_channels, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = x.view(-1, 256, 6, 6)
        
        x = self.conv_transpose1(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        x = self.conv_transpose2(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        x = self.conv_transpose3(x)
        x = self.tanh(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, image_size, image_channels):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(0.4)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(0.4)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.leaky_relu3 = nn.LeakyReLU(0.2)
        self.dropout3 = nn.Dropout(0.4)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.dropout3(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/attributes.csv", help="Path to dataset (attributes.csv)")
    parser.add_argument("--images_path", default="./data/images/", help="Path to images")
    parser.add_argument("--model_output_path", default="./models/", help="Path to output the generator model")
    parser.add_argument("--images_output_path", default="./gen_images/", help="Path to output generated images during training")
    parser.add_argument("--codings_size", type=int, default=100, help="Size of the latent z vector")
    parser.add_argument("--image_size", type=int, default=24, help="Images size")
    parser.add_argument("--image_channels", type=int, default=4, help="Images channels")
    parser.add_argument("--batch_size", type=int, default=16, help="Input batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    args = parser.parse_args()
    print(args)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.images_output_path and (os.path.exists(args.images_output_path) == False):
        print(f"Saving generated images during training at: {args.images_output_path}")
        os.mkdir(args.images_output_path)

    print("Loading the dataset...")
    df = pd.read_csv(args.data_path)
    df.id = df.id.apply(lambda x: f"{args.images_path}punk{x:03d}.png")

    print("Creating PyTorch DataLoader...")
    dataloader = get_dataloader({"paths": df.id}, args.batch_size, args.image_size, args.image_channels)
    
    generator = Generator(args.codings_size, args.image_size, args.image_channels).to(device)
    print("Generator architecture:")
    print(generator)

    discriminator = Discriminator(args.image_size, args.image_channels).to(device)
    print("Discriminator architecture:")
    print(discriminator)

    gen_optimizer = optim.RMSprop(generator.parameters(), lr=0.001)
    disc_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print("Training model...")
    train(dataloader, args.epochs, args.batch_size, args.codings_size, generator, discriminator, 
          gen_optimizer, disc_optimizer, criterion, device)

    print(f"Saving model at: {args.model_output_path}...")
    os.makedirs(args.model_output_path, exist_ok=True)
    model_path = args.model_output_path if args.model_output_path.endswith('.safetensors') else os.path.join(args.model_output_path, 'generator_model.safetensors')
    
    # Save the generator model in safetensors format
    # Metadata is stored as strings in safetensors
    metadata = {
        'codings_size': str(args.codings_size),
        'image_size': str(args.image_size),
        'image_channels': str(args.image_channels)
    }
    save_file(generator.state_dict(), model_path, metadata=metadata)
    print(f"Model saved to: {model_path}")