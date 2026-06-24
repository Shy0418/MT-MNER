"""
MT-MNER: Dataset and Data Processing
Adapted for Twitter-2015/2017 format (IMGID + token\tBIO_label)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from PIL import Image
import numpy as np
import os
import re
from typing import List, Dict, Optional, Tuple
from config import Config


def parse_twitter_file(file_path: str) -> List[Dict]:
    """
    Parse Twitter-2015/2017 format data file.
    
    Format:
        IMGID:17_06_12483
        New     O
        Post    O
        :       O
        Blackburn       B-MISC
        Festival        I-MISC
        of      I-MISC
        Voice   I-MISC
        2017    O
        (empty line)
        IMGID:17_06_2932
        ...
    
    Returns:
        List of dicts with keys: 'text', 'image_id', 'entities', 'tokens', 'labels'
    """
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_img_id = None
    current_tokens = []
    current_labels = []
    
    for line in lines:
        line = line.strip()
        
        # Check for image ID line
        if line.startswith('IMGID:'):
            # Save previous sample if exists
            if current_img_id is not None and current_tokens:
                samples.append({
                    'image_id': current_img_id,
                    'tokens': current_tokens,
                    'labels': current_labels,
                    'text': ' '.join(current_tokens),
                })
            
            # Start new sample
            current_img_id = line.replace('IMGID:', '').strip()
            current_tokens = []
            current_labels = []
        
        elif line == '':
            # Empty line = end of sample
            if current_img_id is not None and current_tokens:
                samples.append({
                    'image_id': current_img_id,
                    'tokens': current_tokens,
                    'labels': current_labels,
                    'text': ' '.join(current_tokens),
                })
            current_img_id = None
            current_tokens = []
            current_labels = []
        
        else:
            # Token and label line
            parts = line.split('\t')
            if len(parts) == 2:
                token, label = parts
                current_tokens.append(token)
                current_labels.append(label)
    
    # Don't forget the last sample
    if current_img_id is not None and current_tokens:
        samples.append({
            'image_id': current_img_id,
            'tokens': current_tokens,
            'labels': current_labels,
            'text': ' '.join(current_tokens),
        })
    
    return samples


class MNERDataset(Dataset):
    """
    Multimodal Named Entity Recognition Dataset
    Supports Twitter-2015/2017 format
    """
    def __init__(
        self,
        config: Config,
        data_path: str,
        image_dir: str,
        tokenizer: AutoTokenizer,
        split: str = 'train',
        max_length: int = 128
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.image_dir = image_dir
        
        # Parse data file
        self.data = parse_twitter_file(data_path)
        
        # Build label mapping
        self.label2id, self.id2label = self._build_label_mapping()
        
        print(f'Loaded {len(self.data)} samples from {data_path}')
    
    def _build_label_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build BIO label mapping"""
        label2id = {'O': 0}
        id2label = {0: 'O'}
        
        idx = 1
        for entity_type in self.config.entity_types:
            label2id[f'B-{entity_type}'] = idx
            id2label[idx] = f'B-{entity_type}'
            idx += 1
            label2id[f'I-{entity_type}'] = idx
            id2label[idx] = f'I-{entity_type}'
            idx += 1
        
        return label2id, id2label
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        text = item['text']
        image_id = item['image_id']
        orig_tokens = item['tokens']
        orig_labels = item['labels']
        
        # Tokenize text with RoBERTa tokenizer
        # RoBERTa uses byte-level BPE, so we need to handle subword tokens
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offset_mapping = encoding['offset_mapping'].squeeze(0)
        
        # Create labels aligned with subword tokens
        labels = self._align_labels(orig_tokens, orig_labels, text, offset_mapping)
        
        # Load image
        image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        pixel_values = self._load_image(image_path)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixel_values': pixel_values,
            'pixel_values_vit': pixel_values,
            'pixel_values_res': pixel_values,
        }
    
    def _align_labels(
        self,
        orig_tokens: List[str],
        orig_labels: List[str],
        text: str,
        offset_mapping: torch.Tensor
    ) -> torch.Tensor:
        """
        Align original BIO labels with RoBERTa subword tokens.
        
        Strategy:
        1. Build character-level label array
        2. For each subword token, look up its character span
        3. Assign the label of the first character of the span
        """
        # Build character-to-label mapping
        char_labels = ['O'] * len(text)
        char_pos = 0
        
        for token, label in zip(orig_tokens, orig_labels):
            # Find the token in text starting from char_pos
            token_start = text.find(token, char_pos)
            if token_start >= 0:
                token_end = token_start + len(token)
                for pos in range(token_start, token_end):
                    if pos < len(char_labels):
                        char_labels[pos] = label
                char_pos = token_end
        
        # Create labels for subword tokens
        labels = torch.full((self.max_length,), -100, dtype=torch.long)
        
        for i in range(min(len(offset_mapping), self.max_length)):
            start, end = offset_mapping[i]
            
            # Skip special tokens (start=0, end=0)
            if start == 0 and end == 0:
                continue
            
            # Get the label of the first character of this token
            if start < len(char_labels):
                label_str = char_labels[start]
                if label_str in self.label2id:
                    labels[i] = self.label2id[label_str]
        
        return labels
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        try:
            if not os.path.exists(image_path):
                # Try alternative paths
                alt_path = image_path.replace('.jpg', '.png')
                if os.path.exists(alt_path):
                    image_path = alt_path
                else:
                    # Return blank image
                    return torch.zeros((3, 224, 224))
            
            image = Image.open(image_path).convert('RGB')
            # Resize to 224x224
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            # Convert to tensor and normalize
            pixel_values = torch.from_numpy(np.array(image)).float()
            pixel_values = pixel_values.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            # Normalize to [0, 1]
            pixel_values = pixel_values / 255.0
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            pixel_values = (pixel_values - mean) / std
        except Exception as e:
            # Return a blank image if loading fails
            pixel_values = torch.zeros((3, 224, 224))
        
        return pixel_values


def create_data_loaders(
    config: Config,
    tokenizer: AutoTokenizer
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, dev, test data loaders for Twitter-2017 format"""
    
    dataset_dir = os.path.join(config.data_dir, config.dataset_name)
    
    train_path = os.path.join(dataset_dir, 'train.txt')
    dev_path = os.path.join(dataset_dir, 'valid.txt')
    test_path = os.path.join(dataset_dir, 'test.txt')
    
    # Image directory (nested structure: twitter2017_images/twitter2017_images/)
    image_dir = os.path.join(dataset_dir, 'twitter2017_images', 'twitter2017_images')
    
    # Check if images exist directly
    if not os.path.exists(image_dir):
        image_dir = os.path.join(dataset_dir, 'twitter2017_images')
    
    print(f'Data directory: {dataset_dir}')
    print(f'Image directory: {image_dir}')
    print(f'Train file: {train_path}')
    print(f'Dev file: {dev_path}')
    print(f'Test file: {test_path}')
    
    train_dataset = MNERDataset(
        config, train_path, image_dir, tokenizer,
        split='train', max_length=config.text_max_length
    )
    dev_dataset = MNERDataset(
        config, dev_path, image_dir, tokenizer,
        split='dev', max_length=config.text_max_length
    )
    test_dataset = MNERDataset(
        config, test_path, image_dir, tokenizer,
        split='test', max_length=config.text_max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, dev_loader, test_loader