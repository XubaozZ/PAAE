
# Adaptive Feature Alignment and Enhancement for Precise Fine-Grained Visual Recognition

This repository implements a Fine-Grained Visual Classification (FGVC) network using PAAE (Part-Aware feature Alignment and Enhancement) and other associated modules. The model is designed to perform feature enhancement and extraction from images, specifically addressing challenges like fine-grained recognition and background complexity.

## Dependencies and Installation

python == 3.12
PyTorch == 2.6.0+cu124
torchvision == 0.21.0+cu124
scikit-image == 0.25.2
timm == 1.0.15
numpy == 2.3.2

## Key Algorithms

### 1. PAAE

The PAAE model is designed to effectively extract discriminative features. Our approach comprises a Progressive Part Mining Module for capturing discriminative features across different layers, an Adaptive Scale Displacement Alignment Module for addressing feature space misalignment, and a Dual-Path Feature Enhancement Module for highlighting foreground features.

### 2. Key Modules

A Progressive Part Mining Module (PPMiner) that captures shape-aware cues with minimal background noise, demonstrating strong robustness to small and localized discriminative regions.
An Adaptive Scale Displacement Alignment (ASDA) Module that introduces learnable scale and positional biases for more accurate multi-scale feature alignment under small-object scenarios.
A Dual-Path Feature Enhancement (DPFE) Module that integrates attention-guided foreground enhancement, background suppression, and a multi-head attention diversity loss to preserve subtle but critical visual details.

### 3. Training Process

The model is trained with cross-entropy loss and diversity regularization loss.
The optimizer uses AdamW, with learning rate decay scheduled through a custom scheduler.
Data augmentation like mixup can be applied to improve generalization.

### 4. Performance Evaluation

The performance of the model is evaluated on several benchmark datasets, including:
CUB-200-2011 for bird species classification.
Stanford Dogs dataset for dog breeds classification.

## Contributing

Feel free to fork this repository and submit pull requests for improvements, bug fixes, or new features. Contributions are welcome!

