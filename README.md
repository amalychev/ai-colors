# 🎨 AI Colors - Neural Network for Color Palette Generation

An intelligent system for creating harmonious color palettes based on neural networks. The project lets you train a model on your own images and generate beautiful color combinations.

## ✨ Features

- 🧠 **Training on custom data**: Train the model on your own design images
- 🎯 **Flexible generation**: Provide 1 to 9 input colors and get 2 to 10 colors in the palette
- 🔍 **Color extraction**: Automatic extraction of color palettes from images
- 🌈 **Harmony scoring**: Automatic evaluation of how pleasing a color combination is
- 🚀 **REST API**: Simple HTTP API for integration into other projects
- 🎨 **Multiple methods**: K-means, ColorThief, histogram analysis

## 🛠 Installation

### Requirements
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Installing dependencies

```bash
pip install -r requirements.txt
```

## 📁 Project structure

```
ai-colors/
├── src/                    # Source code
│   ├── color_extractor.py  # Color extraction module
│   ├── neural_network.py   # Neural network architecture
│   ├── trainer.py          # Training system
│   └── api.py             # REST API
├── training_images/        # Folder for training images
├── data/                   # Processed data
├── models/                 # Saved models
├── notebooks/              # Jupyter notebooks
├── tests/                  # Tests
├── main.py                 # Main script
└── requirements.txt        # Dependencies
```

## 🚀 Quick start

### 1. Preparing data

Place your design images into the `training_images/` folder:

```bash
# Create the folder for images
mkdir -p training_images

# Copy your images into this folder
cp /path/to/your/images/* training_images/
```

### 2. Processing images

Extract color palettes from images:

```bash
python main.py process training_images
```

### 3. Training the model

```bash
python main.py train training_images --epochs 100 --batch-size 32
```

### 4. Running the API

```bash
python main.py api
```

The API will be available at: http://localhost:5000

## 📖 Usage

### Command line

#### Extracting colors from an image

```bash
# Extract 5 colors using the K-means method
python main.py extract path/to/image.jpg --colors 5 --method kmeans

# Use ColorThief
python main.py extract path/to/image.jpg --colors 8 --method colorthief
```

#### Testing generation

```bash
# Generate a palette from red and green
python main.py test 255,0,0 0,255,0 --count 7
```

### HTTP API

#### Generating a palette

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input_colors": [[255, 0, 0], [0, 255, 0]],
    "target_count": 5
  }'
```

#### Harmony evaluation

```bash
curl -X POST http://localhost:5000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "colors": [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
  }'
```

#### Random palette

```bash
curl http://localhost:5000/random
```

### Python API

```python
from src.neural_network import ColorPaletteGenerator
from src.trainer import ColorPaletteTrainer

# Load the trained model
trainer = ColorPaletteTrainer()
trainer.load_model("models/color_palette_generator_final.pth")

# Generate a palette
input_colors = [(255, 0, 0), (0, 255, 0)]  # Red and green
palette = trainer.generator.generate_palette(input_colors, target_count=5)

print("Generated palette:", palette)
```

## 🎨 Palette examples

### Input colors: Red + Blue
- 🔴 `(255, 0, 0)` - Red
- 🔵 `(0, 0, 255)` - Blue

### Generated palette:
- 🔴 `(255, 0, 0)` - Red
- 🔵 `(0, 0, 255)` - Blue
- 🟣 `(128, 0, 128)` - Purple
- 🌸 `(255, 192, 203)` - Pink
- 💙 `(173, 216, 230)` - Light blue

**Harmony score: 0.827** ⭐

## ⚙️ Configuration

### Training parameters

- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- The model is automatically saved every 10 epochs

### Neural network architecture

- **Encoder-Decoder** architecture with an attention mechanism
- **Adversarial training** with a discriminator for harmony scoring
- **Custom loss function** accounting for color similarity and diversity

## 📊 Quality metrics

The system evaluates palettes based on the following criteria:

- **Color harmony**: Classic rules (complementary, triadic, analogous)
- **Diversity**: Sufficient distinction between colors
- **Aesthetic appeal**: Based on the trained model

### Rating scale:
- **0.8+** - Excellent 🌟
- **0.6-0.8** - Good ✅
- **0.4-0.6** - Fair ⚠️
- **<0.4** - Poor ❌

## 🔧 Extending

### Adding new color extraction methods

```python
# In color_extractor.py
def custom_extraction_method(self, image_path: str, n_colors: int) -> List[Tuple[int, int, int]]:
    # Your color extraction code
    return colors
```

### Custom loss function

```python
# In neural_network.py
class CustomLoss(nn.Module):
    def forward(self, generated, target):
        # Your loss computation logic
        return loss
```

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

MIT License - see the [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- The PyTorch team for a great framework
- ColorThief for color extraction algorithms
- The design community for inspiration

---

**Made with ❤️ for designers and developers**
