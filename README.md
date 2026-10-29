# Pretrained VS Custom CNN Diagnosis

**Task**: Car body type classification
**Models Compared**: Custom CNN, [Resnet18]([url](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)) with un/frozen backbone
**Dataset**: [Car Body Types Dataset]([url](https://www.kaggle.com/datasets/ademboukhris/cars-body-type-cropped)) (7550 images), 70/15/15 split.

Parameter | Custom CNN | Resnet18_Frozen | Resnet18 
--- | --- | --- | --- 
Backbone | 4 Conv. Blocks | Frozen | Unfrozen Layer 4
--- | --- | --- | --- 
Class. Head | Linear (128->7) | Linear(512->7) | Linear(512->7)
--- | --- | --- | --- 
N Parameters | ~400k | ~3.5k | ~11.8m
--- | --- | --- | --- 
Learning Rate | Red. On Plateau (1e-4) | Constant (1e-4) | Layer 4: 1e-4, FC: 1e-3
--- | --- | --- | --- 
N Epochs | 100 | 10 | 10

## Custom CNN Settings
**Sequential CNN Layers**: (3 -> 32 -> 64 -> 128 -> 256 features), 
**Head**: (256 -> 128)
**Kernel size**: 3x3, 
**Optimizer**: AdamW
**N Epochs**: 100, 
**Lr**: 1e-4, 
**ReduceLROnPlateau**: _patience_ - 8, _mode_ - min, _scale_ - 0.2
**Batch size**: 64, 
**Weight Decay**: 1e-3, 
**Dropout Probability**: 0.1, 
