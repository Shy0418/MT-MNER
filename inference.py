"""
MT-MNER: Inference Script
Adapted for Twitter-2017 dataset format
"""

import torch
from transformers import AutoTokenizer
from PIL import Image
import numpy as np
import os
from typing import List, Dict, Optional, Tuple

from config import Config
from model import MTMNER
from dataset import parse_twitter_file


class MT_MNER_Inference:
    """
    Inference wrapper for MT-MNER model
    """
    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Config] = None,
        device: str = 'cuda'
    ):
        if config is None:
            config = Config()
        
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_name)
        
        # Initialize model
        self.model = MTMNER(
            num_tags=config.num_labels,
            d=config.fusion_hidden_size,
            n_heads=config.cam_num_heads,
            proj_dim=config.proj_dim,
            roberta_name=config.text_encoder_name,
            vit_name=config.vit_model_name,
            freeze_text_layers=config.text_freeze_layers,
            temperature=config.contrastive_temperature,
            top_k=config.top_k,
        )
        
        # Load checkpoint (state_dict only, saved by train.py)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Build label mapping
        self.id2label = self._build_id2label()
        
        print(f'Model loaded from {checkpoint_path}')
        print(f'Using device: {self.device}')
    
    def _build_id2label(self) -> Dict[int, str]:
        """Build ID to label mapping"""
        id2label = {0: 'O'}
        idx = 1
        for entity_type in self.config.entity_types:
            id2label[idx] = f'B-{entity_type}'
            idx += 1
            id2label[idx] = f'I-{entity_type}'
            idx += 1
        return id2label
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess text input"""
        encoding = self.tokenizer(
            text,
            max_length=self.config.text_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
        }
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image input"""
        try:
            if not os.path.exists(image_path):
                print(f'Warning: Image not found at {image_path}')
                return torch.zeros((1, 3, 224, 224))
            
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            pixel_values = torch.from_numpy(np.array(image)).float()
            pixel_values = pixel_values.permute(2, 0, 1)
            pixel_values = pixel_values / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            pixel_values = (pixel_values - mean) / std
            pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f'Error loading image {image_path}: {e}')
            pixel_values = torch.zeros((1, 3, 224, 224))
        
        return pixel_values
    
    def predict(self, text: str, image_path: str) -> List[Dict]:
        """
        Predict entities from text and image
        
        Args:
            text: Input text
            image_path: Path to the image file
        
        Returns:
            List of predicted entities with type and position
        """
        # Preprocess
        text_inputs = self.preprocess_text(text)
        pixel_values = self.preprocess_image(image_path)
        
        # Move to device
        input_ids = text_inputs['input_ids'].to(self.device)
        attention_mask = text_inputs['attention_mask'].to(self.device)
        pixel_values = pixel_values.to(self.device)
        
        # Predict (new model expects vit + res image inputs)
        with torch.no_grad():
            predictions = self.model.predict(
                input_ids, attention_mask, pixel_values, pixel_values
            )
        
        # Decode predictions
        pred_labels = predictions[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        
        # Extract entities
        entities = self._decode_entities(pred_labels, tokens)
        
        return entities
    
    def _decode_entities(
        self,
        labels: np.ndarray,
        tokens: List[str]
    ) -> List[Dict]:
        """Decode predicted labels into entity list"""
        entities = []
        i = 0
        
        while i < len(labels):
            label = self.id2label.get(int(labels[i]), 'O')
            
            if label.startswith('B-'):
                entity_type = label[2:]
                start = i
                entity_tokens = []
                
                # Collect all tokens for this entity
                while i < len(labels):
                    current_label = self.id2label.get(int(labels[i]), 'O')
                    if current_label == f'B-{entity_type}' or current_label == f'I-{entity_type}':
                        entity_tokens.append(tokens[i])
                        i += 1
                    else:
                        break
                
                end = i - 1
                entity_text = self.tokenizer.convert_tokens_to_string(entity_tokens)
                
                entities.append({
                    'type': entity_type,
                    'text': entity_text,
                    'start': start,
                    'end': end,
                })
            else:
                i += 1
        
        return entities
    
    def predict_from_file(self, txt_path: str, image_dir: str) -> List[Dict]:
        """
        Predict entities from a Twitter-format text file
        
        Args:
            txt_path: Path to .txt file (e.g., test.txt)
            image_dir: Directory containing images
        
        Returns:
            List of predictions for each sample
        """
        samples = parse_twitter_file(txt_path)
        results = []
        
        for sample in samples:
            text = sample['text']
            image_id = sample['image_id']
            image_path = os.path.join(image_dir, f'{image_id}.jpg')
            
            entities = self.predict(text, image_path)
            
            results.append({
                'image_id': image_id,
                'text': text,
                'predicted_entities': entities,
                'true_labels': sample['labels'],
            })
        
        return results
    
    def get_aligned_features(self, text: str, image_path: str) -> Dict[str, np.ndarray]:
        """
        Get aligned features for analysis (t-SNE visualization, etc.)
        """
        text_inputs = self.preprocess_text(text)
        pixel_values = self.preprocess_image(image_path)
        
        input_ids = text_inputs['input_ids'].to(self.device)
        attention_mask = text_inputs['attention_mask'].to(self.device)
        pixel_values = pixel_values.to(self.device)
        
        with torch.no_grad():
            features = self.model.get_contrastive_features(
                input_ids, attention_mask, pixel_values, pixel_values
            )
        
        # Convert to numpy
        result = {}
        for key, value in features.items():
            result[key] = value.cpu().numpy()
        
        return result


def demo_inference():
    """
    Demo: Run inference on a sample from Twitter-2017
    """
    # Configuration
    checkpoint_path = './outputs/best_model.pt'
    config = Config()
    
    # Initialize inference
    inferencer = MT_MNER_Inference(checkpoint_path, config)
    
    # Sample from Twitter-2017
    text = "NBA : Kawhi Leonard wins the fan vote"
    image_path = '../datasets/Twitter-17/twitter2017_images/twitter2017_images/17_06_2932.jpg'
    
    # Predict
    entities = inferencer.predict(text, image_path)
    
    # Print results
    print(f'\nInput Text: {text}')
    print(f'Image: {image_path}')
    print('\nPredicted Entities:')
    for entity in entities:
        print(f'  [{entity["type"]}] {entity["text"]} (positions {entity["start"]}-{entity["end"]})')
    
    # Get aligned features
    features = inferencer.get_aligned_features(text, image_path)
    print(f'\nFeature shapes:')
    for key, value in features.items():
        print(f'  {key}: {value.shape}')


if __name__ == '__main__':
    demo_inference()