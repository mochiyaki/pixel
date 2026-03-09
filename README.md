# Pixel

A PyTorch-based Generative Adversarial Network (GAN) for training and generating pixel art images.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Workflow](#workflow)
- [Installation](#installation)
- [User Guide](#user-guide)
- [Configuration](#configuration)
- [Examples](#examples)

---

## Overview

This project implements a Deep Convolutional GAN (DCGAN) to generate pixel art images. It includes:

- **Training Pipeline**: Train a GAN model on your custom image dataset
- **Image Generation**: Generate new images using trained models
- **Interactive GUI**: Tkinter-based interface for real-time generation
- **GGUF Support**: Convert and use GGUF quantized models

---

## Project Structure

```
ai-picture-model-trainer/
│
├── trainer.py                  # GAN training script
├── generator.py                # Image generation (GUI + CLI)
│
├── data/                       # Training data
│   ├── attributes.csv          # Dataset metadata
│   └── images/                 # Training images (punk000.png, punk001.png, ...)
│
├── models/                     # Trained model storage
│   └── generator_model.safetensors
│
├── generated/                  # Generated image outputs
│   ├── output.png              # Grid visualizations
│   └── individual/             # Individual generated images
│
├── gen_images/                 # Training progress images
│   ├── epoch_0.png
│   ├── epoch_1.png
│   └── ...
│
└── gguf/                       # GGUF format support
    └── generator.py            # GGUF model converter/generator
```

### Directory Purpose

| Directory | Purpose |
|-----------|---------|
| `data/` | Input training images and metadata |
| `models/` | Saved trained models (.safetensors format) |
| `generated/` | Output from generator.py |
| `gen_images/` | Training progress visualizations |
| `gguf/` | GGUF quantized model support |

---

## Architecture

### GAN Components

```
┌─────────────────────────────────────────────────────────────┐
│                      GAN Architecture                       │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐                    ┌──────────────────┐
│   Generator      │                    │  Discriminator   │
│                  │                    │                  │
│  Input: Noise    │                    │  Input: Images   │
│  (100-dim)       │                    │  (24x24x4)       │
│                  │                    │                  │
│  ┌────────────┐  │                    │  ┌────────────┐  │
│  │   FC       │  │                    │  │   Conv2D   │  │
│  │  (9,216)   │  │                    │  │  64 filters│  │
│  └────────────┘  │                    │  └────────────┘  │
│        ↓         │                    │        ↓         │
│  ┌────────────┐  │                    │  ┌────────────┐  │
│  │ Reshape    │  │    ┌──────────┐    │  │   Conv2D   │  │
│  │ (256,6,6)  │  │───→│  Real or │←───│  │ 128 filters│  │
│  └────────────┘  │    │   Fake?  │    │  └────────────┘  │
│        ↓         │    └──────────┘    │        ↓         │
│  ┌────────────┐  │                    │  ┌────────────┐  │
│  │ ConvTrans  │  │                    │  │   Conv2D   │  │
│  │ 128 filters│  │                    │  │ 256 filters│  │
│  └────────────┘  │                    │  └────────────┘  │
│        ↓         │                    │        ↓         │
│  ┌────────────┐  │                    │  ┌────────────┐  │
│  │ ConvTrans  │  │                    │  │  GlobalAvg │  │
│  │  64 filters│  │                    │  │    Pool    │  │
│  └────────────┘  │                    │  └────────────┘  │
│        ↓         │                    │        ↓         │
│  ┌────────────┐  │                    │  ┌────────────┐  │
│  │ ConvTrans  │  │                    │  │  FC + Sig  │  │
│  │  4 channels│  │                    │  │  (0-1)     │  │
│  └────────────┘  │                    │  └────────────┘  │
│                  │                    │                  │
│  Output: Image   │                    │  Output: Score   │
│  (24x24x4 RGBA)  │                    │  (Real/Fake)     │
└──────────────────┘                    └──────────────────┘
```

### Model Details

**Generator**:
- Input: 100-dimensional latent vector (random noise)
- Architecture: FC → BatchNorm → 3x ConvTranspose2D with BatchNorm
- Output: 24x24x4 RGBA image (values in [-1, 1])
- Activation: LeakyReLU + Tanh (output)

**Discriminator**:
- Input: 24x24x4 RGBA image
- Architecture: 3x Conv2D with Dropout → GlobalAvgPool → FC
- Output: Probability score [0, 1] (real vs. fake)
- Activation: LeakyReLU + Sigmoid (output)

---

## Workflow

### Training Workflow

```
┌─────────────┐
│  Start      │
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│  Load Dataset        │
│  (data/images/)      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Initialize Models   │
│  - Generator         │
│  - Discriminator     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Training Loop       │◄─────────┐
│  (N epochs)          │          │
└──────┬───────────────┘          │
       │                          │
       ▼                          │
┌──────────────────────┐          │
│  For each batch:     │          │
│  1. Train Discrim.   │          │
│  2. Train Generator  │          │
└──────┬───────────────┘          │
       │                          │
       ▼                          │
┌──────────────────────┐          │
│  Save Progress       │          │
│  (gen_images/)       │──────────┘
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Save Final Model    │
│ (models/*.safetensors)
└──────┬───────────────┘
       │
       ▼
┌──────────────┐
│  Complete    │
└──────────────┘
```

### Generation Workflow

```
┌─────────────────────┐
│  Mode Selection     │
└──────┬──────────────┘
       │
       ├─────────────────────────┐
       │                         │
       ▼                         ▼
┌──────────────┐       ┌──────────────────┐
│  GUI Mode    │       │   CLI Mode       │
└──────┬───────┘       └──────┬───────────┘
       │                      │
       ▼                      ▼
┌──────────────┐       ┌──────────────────┐
│ Load Model   │       │ Load Model       │
│ (safetensors)│       │ Parse Args       │
└──────┬───────┘       └──────┬───────────┘
       │                      │
       ▼                      ▼
┌──────────────┐       ┌──────────────────┐
│ Tkinter GUI  │       │ Generate N Images│
│ - Button 1x1 │       │ - Custom grid    │
│ - Button 3x3 │       │ - Custom seed    │
│ - Button 5x5 │       │ - Save options   │
└──────┬───────┘       └──────┬───────────┘
       │                      │
       ▼                      ▼
┌──────────────┐       ┌──────────────────┐
│ On Click:    │       │ Save Grid        │
│ Generate     │       │ Save Individual  │
│ Display      │       │ (optional)       │
└──────┬───────┘       └──────────────────┘
       │
       ▼
┌──────────────┐
│ Interactive  │
│ Generation   │
└──────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone/Download the repository**

2. **Install dependencies**:

```bash
pip install torch torchvision
pip install numpy pandas matplotlib pillow
pip install safetensors
```

3. **Prepare your dataset**:

Place your training images in `data/images/` with filenames like `punk000.png`, `punk001.png`, etc.

Create `data/attributes.csv`:
```csv
id
0
1
2
...
```
*this is a community dataset, serves as an example or demo, we do not tier to crytopunk project, you could replace it with your own dataset (see details under Examples session)

---

## User Guide

### 1. Training a Model

Train the GAN on your dataset:

```bash
python trainer.py \
    --data_path ./data/attributes.csv \
    --images_path ./data/images/ \
    --model_output_path ./models/ \
    --images_output_path ./gen_images/ \
    --epochs 50 \
    --batch_size 16
```

**Training Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_path` | `./data/attributes.csv` | Path to dataset metadata |
| `--images_path` | `./data/images/` | Directory containing training images |
| `--model_output_path` | `./models/` | Where to save trained model |
| `--images_output_path` | `./gen_images/` | Save progress images during training |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `16` | Training batch size |
| `--codings_size` | `100` | Latent vector dimension |
| `--image_size` | `24` | Output image size (24x24) |
| `--image_channels` | `4` | Image channels (4=RGBA, 3=RGB) |

**Training Output**:
- Progress displayed: `Epoch X/Y - Gen Loss: X.XXXX, Disc Loss: X.XXXX`
- Progress images saved to `gen_images/epoch_N.png`
- Final model saved to `models/generator_model.safetensors`

---

### 2. Generating Images

#### Option A: Interactive GUI Mode (Default)

Launch the Tkinter GUI for real-time generation:

```bash
python generator.py
```

or explicitly:

```bash
python generator.py --gui
```

**GUI Controls**:
- **Generate 1 avatar**: Single image
- **Generate 3x3 avatars**: 3x3 grid (9 images)
- **Generate 5x5 avatars**: 5x5 grid (25 images)
- **Terminate**: Close the application

---

#### Option B: Command-Line Interface (CLI)

Batch generate images from terminal:

**Basic generation** (16 images):
```bash
python generator.py --num_images 16 --output_path ./generated/output.png
```

**Custom grid** (8x8 = 64 images):
```bash
python generator.py --grid_size 8 --output_path ./generated/grid_8x8.png
```

**Reproducible generation** (with seed):
```bash
python generator.py --grid_size 4 --seed 42 --output_path ./generated/seed42.png
```

**Save individual images**:
```bash
python generator.py \
    --num_images 100 \
    --save_individual \
    --individual_output_dir ./generated/individual/ \
    --output_path ./generated/batch.png
```

**CLI Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | `./models/generator_model.safetensors` | Path to trained model |
| `--output_path` | `./generated/output.png` | Output path for grid image |
| `--num_images` | `16` | Number of images to generate |
| `--grid_size` | `None` | Grid size N for NxN layout |
| `--seed` | `None` | Random seed for reproducibility |
| `--save_individual` | `False` | Save each image separately |
| `--individual_output_dir` | `./generated/individual/` | Directory for individual images |

---

### 3. GGUF Model Support

Use quantized GGUF models for smaller file sizes:

```bash
cd gguf/
python generator.py
```

The GGUF generator will:
1. Detect available `.gguf` files in the directory
2. Prompt you to select a model
3. Convert GGUF → SafeTensors format
4. Launch the standard generator

---

## Configuration

### Model Architecture Configuration

Modify these parameters in both `trainer.py` and `generator.py`:

```python
--codings_size 100        # Latent vector dimension
--image_size 24           # Output image size
--image_channels 4        # RGBA (4) or RGB (3)
```

### Training Hyperparameters

In `trainer.py`:

```python
# Optimizer
gen_optimizer = optim.RMSprop(generator.parameters(), lr=0.001)
disc_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.001)

# Loss function
criterion = nn.BCELoss()

# Dropout rate (in Discriminator)
nn.Dropout(0.4)
```

### Data Preprocessing

In `trainer.py` → `ImageDataset`:

```python
transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * channels, [0.5] * channels)  # [-1, 1]
])
```

---

## Examples

### Example 1: Train on Custom Dataset

```bash
# Prepare your data
# data/images/punk000.png, punk001.png, ..., punk099.png
# data/attributes.csv with ids 0-99

# Train for 100 epochs
python trainer.py \
    --data_path ./data/attributes.csv \
    --images_path ./data/images/ \
    --epochs 100 \
    --batch_size 32 \
    --model_output_path ./models/my_model.safetensors
```

### Example 2: Generate with Specific Seed

```bash
# Generate same images every time
python generator.py \
    --model_path ./models/generator_model.safetensors \
    --grid_size 5 \
    --seed 12345 \
    --output_path ./results/reproducible.png
```

### Example 3: Batch Generation

```bash
# Generate 1000 individual images
python generator.py \
    --num_images 1000 \
    --save_individual \
    --individual_output_dir ./dataset_synthetic/ \
    --output_path ./dataset_synthetic/overview.png
```

### Example 4: Monitor Training Progress

```bash
# Training with progress visualization
python trainer.py \
    --epochs 200 \
    --images_output_path ./training_progress/

# View progress images
ls ./training_progress/
# epoch_0.png, epoch_1.png, ..., epoch_199.png
```

---

## Technical Details

### Model File Format

Models are saved in **SafeTensors** format (`.safetensors`) with embedded metadata:

```python
metadata = {
    'codings_size': '100',
    'image_size': '24',
    'image_channels': '4'
}
```

This ensures the generator automatically loads the correct architecture.

### Image Value Ranges

- **Training**: Images normalized to [-1, 1]
- **Generation output**: Images scaled to [0, 1]
- **Saved files**: Images saved as uint8 [0, 255]

### GPU Support

The code automatically detects and uses CUDA if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## Troubleshooting

**Q: Training loss not decreasing?**
- Try adjusting learning rates
- Increase batch size or epochs
- Check if dataset has sufficient variety

**Q: Generated images look like noise?**
- Model needs more training epochs
- Dataset may be too small (need 50+ images minimum)
- Try adjusting discriminator dropout rate

**Q: GUI not launching?**
- Check Tkinter installation: `python -m tkinter`
- On Linux: `sudo apt-get install python3-tk`

**Q: CUDA out of memory?**
- Reduce batch size: `--batch_size 8`
- Reduce image size: `--image_size 16`

---

## License

This project is provided as-is for educational and creative purposes.

---

## Acknowledgments

- Built with PyTorch
- Inspired by DCGAN architecture
- Uses SafeTensors for model serialization
