"""
MT-MNER: Configuration
Based on the paper: MT-MNER: A Multimodal Named Entity Recognition Model
Based on Multi-View Contrastive Learning and Two-Stage Feature Fusion
"""

import os

class Config:
    # ============ Model Paths (HuggingFace) ============
    # Models are downloaded from HuggingFace and cached locally
    
    # Text encoder: RoBERTa-base (HF)
    text_encoder_name = 'FacebookAI/roberta-base'
    text_max_length = 128
    text_hidden_size = 768
    text_num_attention_heads = 8
    text_num_hidden_layers = 12
    text_freeze_layers = 5  # Freeze first 5 layers
    
    # Global visual encoder: ViT (HF)
    vit_model_name = 'google/vit-base-patch16-224'
    vit_hidden_size = 768
    vit_num_patches = 196  # 14x14 patches for 224x224 image
    
    # ResNet-50 (torchvision auto-download)
    faster_rcnn_num_regions = 16
    faster_rcnn_feature_dim = 2048
    local_visual_hidden_size = 768
    
    # Multi-view Contrastive Learning
    contrastive_temperature = 0.1
    contrastive_batch_size = 128
    proj_dim = 256
    top_k = 5
    lambda_s = 1.0
    
    # Two-stage Feature Fusion
    fusion_hidden_size = 768
    cam_num_heads = 8
    cam_dropout = 0.1
    gfm_dropout = 0.1
    
    # ============ Training ============
    optimizer = 'AdamW'
    learning_rate = 2e-5
    dropout_rate = 0.2
    num_epochs = 20
    train_batch_size = 16  # Reduced for memory
    eval_batch_size = 32
    warmup_ratio = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0
    
    # ============ Paths ============
    output_dir = './outputs'
    checkpoint_dir = './outputs'  # Save best model to outputs/
    log_dir = './logs'
    
    # ============ Dataset ============
    dataset_name = 'Twitter-17'  # twitter2015, twitter2017, wikidiverse
    data_dir = './datasets'
    
    # BIO tagging scheme
    entity_types = ['PER', 'LOC', 'ORG', 'MISC']
    # B-entity, I-entity, O
    num_labels = 2 * len(entity_types) + 1  # 9
    
    # ============ Device ============
    device = 'cuda'  # or 'cpu'
    
    # ============ Seed ============
    seed = 42
    # 42, 1, 2, 3, 4