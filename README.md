# MT-MNER: Multimodal Named Entity Recognition

Implementation of **MT-MNER: A Multimodal Named Entity Recognition Model Based on Multi-View Contrastive Learning and Two-Stage Feature Fusion**.

## Architecture Overview

```
Input: Text + Image
    │
    ├── Feature Extraction
    │   ├── RoBERTa → Text Features
    │   ├── ViT → Local Visual Features
    │   └── ResNet-50 → Global Visual Features
    │
    ├── Multi-view Contrastive Learning
    │   ├── Text ↔ Global visual alignment
    │   └── token ↔ patch alignment
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

