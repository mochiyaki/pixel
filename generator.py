import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import *
from PIL import ImageTk, Image
import random
from safetensors.torch import load_file

# Generator model definition (must match the training architecture)
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

def load_model(model_path, device='cpu'):
    """
    Load the trained generator model from safetensors format.
    
    Args:
        model_path: Path to the .safetensors model file
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Loaded generator model and configuration parameters
    """
    # Load state dict and metadata from safetensors
    state_dict = load_file(model_path)
    
    # Load metadata from safetensors file
    from safetensors import safe_open
    with safe_open(model_path, framework="pt", device=str(device)) as f:
        metadata = f.metadata()
    
    # Extract model configuration from metadata
    codings_size = int(metadata['codings_size'])
    image_size = int(metadata['image_size'])
    image_channels = int(metadata['image_channels'])
    
    # Create generator model
    model = Generator(codings_size, image_size, image_channels)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model configuration: codings_size={codings_size}, image_size={image_size}, image_channels={image_channels}")
    
    return model, codings_size, image_size, image_channels

def generate_images(model, num_images, codings_size=100, seed=None, device='cpu'):
    """
    Generate images using the trained GAN generator model.
    
    Args:
        model: Loaded PyTorch generator model
        num_images: Number of images to generate
        codings_size: Size of the latent vector (default: 100)
        seed: Random seed for reproducibility
        device: Device to run generation on
    
    Returns:
        Generated images as numpy array (scaled to [0, 1])
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate random noise as input
    noise = torch.randn(num_images, codings_size, device=device)
    
    # Generate images
    with torch.no_grad():
        generated_images = model(noise)
    
    # Convert from CHW to HWC format and scale from [-1, 1] to [0, 1]
    generated_images = generated_images.permute(0, 2, 3, 1).cpu().numpy()
    generated_images = (generated_images + 1) / 2  # Scale to [0, 1]
    
    return generated_images

def save_image_grid(images, output_path, grid_size=None):
    """
    Save generated images as a grid visualization.
    
    Args:
        images: Array of generated images
        output_path: Path to save the grid image
        grid_size: Optional grid size (rows, cols). If None, auto-calculate square grid
    """
    num_images = images.shape[0]
    
    if grid_size is None:
        # Auto-calculate square grid
        grid_rows = int(np.sqrt(num_images))
        grid_cols = int(np.ceil(num_images / grid_rows))
    else:
        grid_rows, grid_cols = grid_size
    
    fig = plt.figure(figsize=(grid_cols * 2, grid_rows * 2))
    
    for i in range(min(num_images, grid_rows * grid_cols)):
        plt.subplot(grid_rows, grid_cols, i + 1)
        
        # Handle different image formats
        if images.shape[-1] == 1:
            # Grayscale
            plt.imshow(images[i, :, :, 0], cmap='gray')
        else:
            # RGB or RGBA
            plt.imshow(images[i])
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_individual_images(images, output_dir, prefix="generated"):
    """
    Save each generated image as a separate file.
    
    Args:
        images: Array of generated images
        output_dir: Directory to save individual images
        prefix: Prefix for image filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img in enumerate(images):
        # Convert to uint8 format (0-255)
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Save using matplotlib to handle RGBA correctly
        output_path = output_dir / f"{prefix}_{i:04d}.png"
        plt.imsave(output_path, img_uint8)
    
    print(f"Saved {len(images)} individual images to: {output_dir}")


# ============ TKINTER UI MODE ============

def run_gui(model_path, output_path):
    """
    Run Tkinter GUI for interactive image generation.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model once at startup
    print(f"Loading model from: {model_path}")
    try:
        model, codings_size, image_size, image_channels = load_model(model_path, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Tkinter window
    root = Tk()
    root.title("CryptoPunk Generator")
    root.columnconfigure([0, 1, 2, 3], minsize=200)
    
    # Create a placeholder image if output doesn't exist
    if not os.path.exists(output_path):
        fig = plt.figure(figsize=(4, 4))
        plt.text(0.5, 0.5, 'Click a button to generate!', 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Load and display initial image
    img = ImageTk.PhotoImage(Image.open(output_path))
    panel = Label(root, image=img)
    panel.grid(row=1, columnspan=4, sticky="nsew")
    
    def update_img():
        """Update the displayed image"""
        new_img = ImageTk.PhotoImage(Image.open(output_path))
        panel.configure(image=new_img)
        panel.image = new_img
    
    def generate(grid_size):
        """Generate images in a grid"""
        print(f"Generating {grid_size}x{grid_size} grid...")
        n_img = grid_size * grid_size
        seed = random.getrandbits(32)
        
        # Generate images
        images = generate_images(model, n_img, codings_size, seed, device)
        
        # Create grid visualization
        fig = plt.figure(figsize=(8, 8))
        for i in range(n_img):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(images[i, :, :, :])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Generated with seed: {seed}")
        update_img()
    
    # Create buttons
    btn_1 = Button(root, text="Generate 1 cryptopunk", command=lambda: generate(1))
    btn_3 = Button(root, text="Generate 3x3 cryptopunks", command=lambda: generate(3))
    btn_5 = Button(root, text="Generate 5x5 cryptopunks", command=lambda: generate(5))
    btn_q = Button(root, text="Terminate", command=root.quit)
    
    btn_1.grid(row=0, column=0, sticky="nsew")
    btn_3.grid(row=0, column=1, sticky="nsew")
    btn_5.grid(row=0, column=2, sticky="nsew")
    btn_q.grid(row=0, column=3, sticky="nsew")
    
    print("\nGUI started! Click buttons to generate images.")
    root.mainloop()


# ============ CLI MODE ============

def run_cli(args):
    """
    Run command-line interface for batch image generation.
    """
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train the model first using trainer.py")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the trained model
    print(f"Loading model from: {args.model_path}")
    try:
        model, codings_size, image_size, image_channels = load_model(args.model_path, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Calculate actual number of images for grid
    if args.grid_size is not None:
        num_images = args.grid_size * args.grid_size
        grid_size = (args.grid_size, args.grid_size)
        print(f"Generating {num_images} images in a {args.grid_size}x{args.grid_size} grid")
    else:
        num_images = args.num_images
        grid_size = None
        print(f"Generating {num_images} images")
    
    # Generate images
    print("Generating images...")
    images = generate_images(model, num_images, codings_size, args.seed, device)
    print(f"Generated images shape: {images.shape}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Create output directory if needed
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save grid visualization
    save_image_grid(images, args.output_path, grid_size)
    print(f"Grid image saved to: {args.output_path}")
    
    # Optionally save individual images
    if args.save_individual:
        save_individual_images(images, args.individual_output_dir)
    
    print("\nGeneration complete!")
    if args.seed is not None:
        print(f"Seed used: {args.seed} (use same seed to reproduce these images)")


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description="Generate images using trained GAN model")
    parser.add_argument("--gui", action="store_true",
                        help="Launch Tkinter GUI interface (default if no other args)")
    parser.add_argument("--model_path", type=str, default="./models/generator_model.safetensors",
                        help="Path to the trained generator model (.safetensors file)")
    parser.add_argument("--output_path", type=str, default="./generated/output.png",
                        help="Path to save the generated image grid")
    parser.add_argument("--num_images", type=int, default=16,
                        help="Number of images to generate (CLI mode, default: 16)")
    parser.add_argument("--grid_size", type=int, default=None,
                        help="Grid size N for NxN layout (CLI mode)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (CLI mode only)")
    parser.add_argument("--save_individual", action="store_true",
                        help="Save each generated image as a separate file (CLI mode)")
    parser.add_argument("--individual_output_dir", type=str, default="./generated/individual/",
                        help="Directory to save individual images (CLI mode)")
    
    args = parser.parse_args()
    
    # Determine mode: GUI if --gui flag or if no CLI-specific args provided
    cli_args_provided = (args.grid_size is not None or 
                         args.num_images != 16 or 
                         args.seed is not None or 
                         args.save_individual)
    
    if args.gui or not cli_args_provided:
        # GUI mode
        run_gui(args.model_path, args.output_path)
    else:
        # CLI mode
        run_cli(args)


if __name__ == "__main__":
    main()
