# MT-MNER: Multimodal Named Entity Recognition

Implementation of **MT-MNER: A Multimodal Named Entity Recognition Model Based on Multi-View Contrastive Learning and Two-Stage Feature Fusion**.

## Architecture Overview

```
Input: Text + Image
    │
    ├── Feature Extraction
    │   ├── RoBERTa → Text Features (f_T)
    │   ├── ViT → Global Visual Features (f_G)
    │   └── Faster R-CNN → Local Visual Features (f_L)
    │
    ├── Multi-view Contrastive Learning
    │   ├── T2G: Text ↔ Global Visual alignment
    │   ├── T2L: Text ↔ Local Visual alignment
    │   └── G2L: Global Visual ↔ Local Visual alignment
    │
    ├── Two-stage Feature Fusion
    │   ├── Hierarchical Fusion (3× CAM + 2× GFM)
    │   └── Global Fusion (Attention + MLP)
    │
    └── Prediction (CRF)
```

## Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

```
MT-MNER/
├── config.py          # Configuration
├── model.py           # Full model implementation
├── dataset.py         # Dataset and data loading
├── train.py           # Training script
├── inference.py       # Inference script
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## Key Components

### 1. Feature Extraction
- **Text**: RoBERTa-base (first 5 layers frozen)
- **Global Visual**: ViT-base-patch16-224 with mean pooling
- **Local Visual**: ResNet-50 backbone with ROI pooling (16 regions)

### 2. Multi-view Contrastive Learning
- Three contrastive pairs: T2G, T2L, G2L
- InfoNCE loss with temperature τ = 0.1
- Projects features into unified semantic space

### 3. Two-stage Feature Fusion
- **Hierarchical Fusion**: 3 Cross-Attention Modules + 2 Gated Fusion Modules
  - Step 1: Surface-aware perception (f₁)
  - Step 2: Semantic alignment (f₂)
  - Step 3: Deep understanding (f₃)
- **Global Fusion**: Attention-based calibration + MLP aggregation

### 4. Prediction
- Linear classifier + CRF decoding
- BIO tagging scheme (PER, LOC, ORG, MISC)

## Usage

### Training

```bash
python train.py
```

### Inference

```python
from inference import MT_MNER_Inference

# Load model
inferencer = MT_MNER_Inference('./checkpoints/best_model.pt')

# Predict
entities = inferencer.predict(
    text="Yes RT @ESPN_FirstTake: Both Skip And Stephen A. are going with FSU...",
    image_path="./data/sample.jpg"
)

print(entities)
# [{'type': 'PER', 'text': 'Skip', ...}, {'type': 'PER', 'text': 'Stephen A.', ...}]
```

## Dataset Format

Expected JSON format for Twitter-2015/2017:

```json
[
    {
        "text": "Yes RT @ESPN_FirstTake: Both Skip And Stephen A. are going with FSU...",
        "image": "sample.jpg",
        "entities": [
            {"name": "Skip", "type": "PER"},
            {"name": "Stephen A.", "type": "PER"},
            {"name": "FSU", "type": "ORG"}
        ]
    }
]
```

## Citation

```bibtex
@article{han2025mtmner,
    title={MT-MNER: A Multimodal Named Entity Recognition Model Based on Multi-View Contrastive Learning and Two-Stage Feature Fusion},
    author={Han, Mingxing and Li, Jiaxuan and Xu, Liwei},
    year={2025}
}