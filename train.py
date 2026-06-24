"""
MT-MNER: Training Script
Adapted for Twitter-2017 dataset format
Models download from HuggingFace (auto-cached), best model saved to outputs/
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import os
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from config import Config
from model import MTMNER
from dataset import MNERDataset, create_data_loaders
from efficiency import (
    measure_all_efficiency,
    print_efficiency_summary,
    reset_gpu_memory_stats,
)

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def compute_metrics(
    predictions: List[int],
    labels: List[int],
    id2label: Dict[int, str]
) -> Dict[str, float]:
    """
    Compute overall and per-class precision, recall, F1 for entity recognition.
    predictions: flat list of predicted label IDs
    labels: flat list of ground truth label IDs
    id2label: mapping
    """
    all_preds = np.array(predictions)
    all_labels = np.array(labels)

    # Token-level Accuracy
    valid_mask = all_labels != -100
    correct = (all_preds[valid_mask] == all_labels[valid_mask]).sum()
    total = valid_mask.sum()
    accuracy = correct / total if total > 0 else 0.0

    # Entity-level Evaluation
    true_entities = extract_entities(all_labels, id2label)
    pred_entities = extract_entities(all_preds, id2label)

    true_positives = len(true_entities & pred_entities)
    false_positives = len(pred_entities - true_entities)
    false_negatives = len(true_entities - pred_entities)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-class Entity Metrics
    true_by_type = defaultdict(set)
    pred_by_type = defaultdict(set)

    for entity in true_entities:
        entity_type, start, end = entity
        true_by_type[entity_type].add(entity)

    for entity in pred_entities:
        entity_type, start, end = entity
        pred_by_type[entity_type].add(entity)

    all_types = set(list(true_by_type.keys()) + list(pred_by_type.keys()))

    per_class_metrics = {}
    for entity_type in sorted(all_types):
        true_set = true_by_type.get(entity_type, set())
        pred_set = pred_by_type.get(entity_type, set())

        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        per_class_metrics[entity_type] = {
            'precision': p * 100,
            'recall': r * 100,
            'f1': f * 100,
            'support': len(true_set),
        }

    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'per_class': per_class_metrics,
    }


def extract_entities(labels: np.ndarray, id2label: Dict[int, str]) -> set:
    """
    Extract entities from BIO tag sequence
    Returns set of (entity_type, start, end) tuples
    """
    entities = set()
    i = 0
    while i < len(labels):
        label = id2label.get(int(labels[i]), 'O')
        if label.startswith('B-'):
            entity_type = label[2:]
            start = i
            i += 1
            while i < len(labels):
                next_label = id2label.get(int(labels[i]), 'O')
                if next_label == f'I-{entity_type}':
                    i += 1
                else:
                    break
            end = i - 1
            entities.add((entity_type, start, end))
        else:
            i += 1
    return entities


def train_epoch(
    model: MTMNER,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    epoch: int,
    lambda_s: float = 1.0
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} Training')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values_vit = batch['pixel_values_vit'].to(device)
        pixel_values_res = batch['pixel_values_res'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_vit=pixel_values_vit,
            pixel_values_res=pixel_values_res,
            labels=labels,
            lambda_s=lambda_s,
        )

        loss = outputs['loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ner': f'{outputs["loss_ner"]:.4f}',
            'coarse': f'{outputs["loss_coarse"]:.4f}',
            'fine': f'{outputs["loss_fine"]:.4f}',
        })

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: MTMNER,
    eval_loader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str]
) -> Dict[str, float]:
    """Evaluate the model"""
    model.eval()

    all_predictions = []  # list of lists (variable length per sample)
    all_labels = []

    for batch in tqdm(eval_loader, desc='Evaluating'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values_vit = batch['pixel_values_vit'].to(device)
        pixel_values_res = batch['pixel_values_res'].to(device)
        labels = batch['labels']

        predictions = model.predict(
            input_ids, attention_mask, pixel_values_vit, pixel_values_res
        )

        # predictions is a list of lists of variable length
        for i, pred_seq in enumerate(predictions):
            label_seq = labels[i]
            # Only evaluate valid positions (not -100)
            valid_mask = (label_seq != -100).cpu().numpy()
            seq_len = len(pred_seq)
            valid_len = min(seq_len, len(valid_mask))
            for j in range(valid_len):
                if valid_mask[j]:
                    all_predictions.append(pred_seq[j])
                    all_labels.append(int(label_seq[j].item()))

    metrics = compute_metrics(
        all_predictions,
        all_labels,
        id2label
    )

    return metrics


def print_metrics(metrics: Dict, title: str = "Results"):
    """Pretty print evaluation metrics"""
    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'{"="*60}')

    print(f'\n  Overall Metrics:')
    print(f'  {"─"*40}')
    print(f'    Accuracy : {metrics["accuracy"]:.2f}%')
    print(f'    Precision: {metrics["precision"]:.2f}%')
    print(f'    Recall   : {metrics["recall"]:.2f}%')
    print(f'    F1       : {metrics["f1"]:.2f}%')

    print(f'\n  Per-Class Metrics:')
    print(f'  {"─"*60}')
    print(f'  {"Class":<10} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Support":>10}')
    print(f'  {"─"*60}')

    for entity_type, class_metrics in sorted(metrics['per_class'].items()):
        print(f'  {entity_type:<10} {class_metrics["precision"]:>8.2f}% '
              f'{class_metrics["recall"]:>8.2f}% {class_metrics["f1"]:>8.2f}% '
              f'{class_metrics["support"]:>8d}')

    print(f'  {"─"*60}')
    print()


def main():
    # Load config
    config = Config()

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    config.device = device
    print(f'Using device: {device}')

    # Set seed
    set_seed(config.seed)

    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Initialize tokenizer from local path
    # local_files_only removed — offline env vars prevent network calls; 
    # the monkey-patched validate_repo_id accepts our local directory path
    print(f'Loading tokenizer from local path: {config.text_encoder_name}')
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_name)

    # Create data loaders
    train_loader, dev_loader, test_loader = create_data_loaders(config, tokenizer)

    # Get label mapping
    dataset_dir = os.path.join(config.data_dir, config.dataset_name)
    train_path = os.path.join(dataset_dir, 'train.txt')
    image_dir = os.path.join(dataset_dir, 'twitter2017_images', 'twitter2017_images')
    if not os.path.exists(image_dir):
        image_dir = os.path.join(dataset_dir, 'twitter2017_images')

    sample_dataset = MNERDataset(
        config, train_path, image_dir, tokenizer, split='train'
    )
    id2label = sample_dataset.id2label

    # Initialize model (downloads from HuggingFace on first run, caches locally)
    print('\nInitializing MT-MNER model (downloading from HuggingFace if needed)...')
    model = MTMNER(
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
    model = model.to(device)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Reset GPU memory stats before training (track peak during training)
    reset_gpu_memory_stats(device)

    # Training loop
    best_f1 = 0.0
    train_losses = []
    eval_metrics_history = []

    print(f'\nStarting training for {config.num_epochs} epochs...')
    print(f'Total steps: {total_steps}, Warmup steps: {warmup_steps}')

    training_start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        print(f'\n{"="*50}')
        print(f'Epoch {epoch}/{config.num_epochs}')
        print(f'{"="*50}')

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            lambda_s=config.lambda_s
        )
        train_losses.append(train_loss)

        # Evaluate on dev set
        dev_metrics = evaluate(model, dev_loader, device, id2label)
        eval_metrics_history.append(dev_metrics)

        print_metrics(dev_metrics, f'Dev Set - Epoch {epoch}')

        # Save best model to outputs/
        if dev_metrics['f1'] > best_f1:
            best_f1 = dev_metrics['f1']
            checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  New best model saved to {checkpoint_path}! F1: {best_f1:.2f}%')

    training_end_time = time.time()
    training_time_s = training_end_time - training_start_time

    # Load best model and evaluate on test set
    print(f'\n{"="*60}')
    print('  Evaluating best model on test set...')
    print(f'{"="*60}')

    checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
    model.load_state_dict(torch.load(checkpoint_path, weights_only=False))

    test_metrics = evaluate(model, test_loader, device, id2label)
    print_metrics(test_metrics, 'Test Set - Final Results')

    # ============ Efficiency Metrics ============
    print(f'\n{"="*60}')
    print('  Measuring efficiency metrics...')
    print(f'{"="*60}')

    efficiency_metrics = measure_all_efficiency(
        model, test_loader, device, training_time_s
    )
    print_efficiency_summary(efficiency_metrics)

    # Save final results
    results = {
        'best_dev_f1': best_f1,
        'test_metrics': test_metrics,
        'train_losses': train_losses,
        'eval_metrics': eval_metrics_history,
        'training_time_s': training_time_s,
        'efficiency': {
            'params': efficiency_metrics['params'],
            'flops': efficiency_metrics['flops'],
            'latency_ms': efficiency_metrics['latency']['avg_latency_ms'],
            'gpu_memory_mb': efficiency_metrics['gpu_memory']['max_allocated_mb'],
        },
        'config': {
            'text_encoder': config.text_encoder_name,
            'vit_model': config.vit_model_name,
            'learning_rate': config.learning_rate,
            'batch_size': config.train_batch_size,
            'num_epochs': config.num_epochs,
            'temperature': config.contrastive_temperature,
            'proj_dim': config.proj_dim,
            'top_k': config.top_k,
            'lambda_s': config.lambda_s,
        }
    }

    results_path = os.path.join(config.output_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\nResults saved to {results_path}')
    print(f'Best model saved to {checkpoint_path}')
    print('Training completed!')


if __name__ == '__main__':
    main()