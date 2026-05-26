# cv2025-homework

Решения домашних работ к курсу [cv2025](https://github.com/tam2511/cv2025) (Computer Vision 2025).

| Файл | Задание | Цель |
|---|---|---|
| [hw3_unet_camvid.ipynb](hw3_unet_camvid.ipynb) | UNet-сегментация на CamVid | mIoU >= 0.7 |
| [hw5_hand_keypoints_3d.ipynb](hw5_hand_keypoints_3d.ipynb) | 3D-регрессия ключевых точек руки (FreiHAND) | 3D MPJPE <= 20 mm |
| [hw7_conditional_gan_mnist.ipynb](hw7_conditional_gan_mnist.ipynb) | Conditional GAN на MNIST | Соответствие меткам |

## Ключевые техники

### hw3 (Lesson 3)
- Энкодер `efficientnet-b3` + декодер UNet с SCSE-attention
- Разрешение 384×384, mixed precision (`16-mixed`)
- Loss: `DiceLoss + CrossEntropy` с `ignore_index=11` (Void)
- AdamW + OneCycleLR (`pct_start=0.1`)
- Аугментации albumentations: HFlip, ShiftScaleRotate, BrightnessContrast, HSV, GaussNoise

### hw5 (Lesson 5)
- Backbone: ResNet18 (ImageNet), глубокая регрессионная голова с Dropout
- Loss: `smooth_l1_loss(beta=0.05)` (Huber) — устойчивее, чем MSE
- AdamW (`wd=1e-4`) + CosineAnnealingLR (`eta_min=1e-7`)

### hw7 (Lesson 7)
- Классический cGAN (Mirza & Osindero): one-hot + concat
- Дискриминатор с тремя членами лосса: real+correct, fake, real+**wrong** label
- `BCEWithLogitsLoss` — численно стабильнее, чем Sigmoid + BCE
