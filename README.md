# Pretrained VS Custom CNN Diagnosis

**Task**: Car body type classification
**Models Compared**: Custom CNN, [Resnet18]([url](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)) with un/frozen backbone
**Dataset**: [Car Body Types Dataset]([url](https://www.kaggle.com/datasets/ademboukhris/cars-body-type-cropped)) (7550 images), 70/15/15 split.

Parameter | Custom CNN | Resnet18_Frozen | Resnet18
| :--- | :--- | :--- | :--- |
**Backbone** | 4 Conv. Blocks | Frozen | Unfrozen Layer 4
**Class. Head** | Linear (128->7) | Linear(512->7) | Linear(512->7)
**N Parameters** | ~400k | ~3.5k | ~11.8m
**Learning Rate** | Red. On Plateau (1e-4) | Constant (1e-4) | Layer 4: 1e-4, FC: 1e-3
**N Epochs** | 100 | 10 | 10

## Custom CNN Settings
    **Sequential CNN Layers**: (3 -> 32 -> 64 -> 128 -> 256 features)
    **Head**: (256 -> 128)
    **Kernel size**: 3x3
    **Optimizer**: AdamW
    **N Epochs**: 100
    **Lr**: 1e-4
    **ReduceLROnPlateau**: _patience_ - 8, _mode_ - min, _scale_ - 0.2
    **Batch size**: 64
    **Weight Decay**: 1e-3
    **Dropout Probability**: 0.1

## Performance Comparison
| Parameter | Custom CNN | Resnet18_Frozen | Resnet18 |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | 79% | 79% | 92% |
| **Training Loss** | 0.53 | 0.56 | 0.17 |
| **Training time** | 2960 sec. | 290 sec. | 304 sec. |
| **Training time Per Epoch** | 29.6 sec. | 29 sec. | 30 sec. |
| **Inference time** | 6.5 ms | 6.97 ms | 7.35 ms |
| **Customizability** | High | Low | Medium |
| **Available Param.** | ~400k | ~3.5k | ~11.8m |

## Confusion Matricies
  _The most commonly confused class is Coupe, hence its similarity to Sedan or Hatchback_
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b3d0c2ac-22f3-4d84-86c8-7f2d9111c3f8" width="200"><br>
      <b>Custom CNN</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/5c340243-dcaa-4126-907c-7055e517da5c" width="200"><br>
      <b>Resnet18_Frozen</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/ba5efa82-7498-48fb-b5c3-c46bdc684984" width="200"><br>
      <b>Resnet18</b>
    </td>
  </tr>
</table>

## Verdict
**Custom CNN Use Cases**: 
  The task is highly specific, so that model pretrained on generic dataset would be of no use (non-natural imagery). 
  When a model requires deep and more complex, or contrary, simple architecture for specific hardware. When full control over model architecture is desired. 
  A good choice when the goal is to learn how CNN architecture works behind the scenes.
**Frozen Backbone Use Cases**: 
  Hardware capacity is severely limited and/or the dataset is extremely small. 
  Can adapt quickly to new datasets, with compromise on accuracy. 
  Since the dense layer is the only one being trained and also is decision making, the number of trainable parameters is limited. 
**Partial Fine-Tuning Use Cases**: 
  For non-specific visual classification tasks. 
  Model can pick up visual clues and patterns on low levels, retrained high levels produce accurate predictions for imagery related to one the low layers were trained on. 
  Works well for where high-level features need domain adaptation. 
  Relatively cheap and fast training process resulting in high accuracy. 
  Limited control over model architecture.
