# Effect of Attention Type and Attention Position in a Pretrained ResNet50: Channel, Spatial, Self-Attention, ECA and CBAM

**CS3807 – Deep Learning Laboratory, Experiments 8, 9 - Attention Mechanisms in a Pretrained ResNet50 on TensorFlow Flowers**

A single pretrained backbone, ResNet50 (ImageNet), used throughout to study how different
attention mechanisms and different insertion positions change classification performance,
complexity, and the internal behaviour of the network, with a controlled comparison of 16 models
and a qualitative analysis based on internal attention maps and Grad-CAM.

## Overview

This project covers the full pipeline end to end:

- Dataset download of the TensorFlow Flowers images (3,670 images, 5 classes), resized to
  $224\times224\times3$ and preprocessed with ResNet50's own `preprocess_input`, with a fixed,
  class-stratified 70/15/15 train/validation/test split (seed 42, 2,569 / 550 / 551 images)
  reused for every experiment
- A baseline pretrained ResNet50 (Global Average Pooling → Dropout → Dense(5)) with the stage
  dimensions verified on the Keras model and four candidate attention positions
  $P_1$–$P_4$ (after conv2_x to conv5_x)
- Channel attention (SE-style, $r=16$), spatial attention, sequential channel→spatial attention,
  lightweight self-attention and a combined module, written as custom Keras layers that never
  change the feature dimensions (recorded before and after every module)
- Experiment 1 (attention type, fixed at $P_3$), Experiment 2 (attention position),
  Experiment 3 (single vs. multiple positions, including the extended $P_2+P_3+P_4$ run, with
  parameters, peak GPU memory, training time and inference time), Experiment 4 (channel→spatial vs.
  spatial→channel) and Experiment 5 (spatial kernel size $k\in\{3,5,7\}$)
- Additional task: ECA and CBAM compared against the other mechanisms under the same protocol
- Two-stage transfer learning (frozen backbone, then fine-tuning conv5_x at a lower learning
  rate), identical for all 16 models, with accuracy, macro precision / recall / F1, trainable and
  total parameters and inference time reported on the test set
- Training/validation accuracy and loss curves, accuracy and macro-F1 bar graphs,
  accuracy-vs-parameters and F1-vs-inference-time scatter plots, and baseline vs. attention
  confusion matrices with per-class analysis
- Qualitative analysis: feature maps before/after attention, channel-attention weights (top-10),
  spatial-attention maps and overlays, self-attention response, Grad-CAM (baseline vs. attention),
  spatial maps at $P_2$, $P_3$, $P_4$, and side-by-side comparisons across mechanisms
- All 35 discussion questions and the 8 attention-comparison questions from the lab manual,
  answered directly in the report
- All 16 figures exported at 600 dpi in Times New Roman (15 pt) as PNG, PDF and EPS, with
  separate downloadable zip archives for each format

## Results

| Model | Position | Accuracy | Macro F1 | Parameters | Time (ms/img) |
|-------|----------|----------|----------|------------|---------------|
| Baseline ResNet50 | – | 94.37% | 0.9436 | 23,597,957 | 7.67 |
| Channel Attention (SE) | $P_3$ | 93.47% | 0.9339 | 23,730,117 | 7.66 |
| Spatial Attention | $P_3$ | 93.65% | 0.9368 | 23,598,056 | 8.42 |
| Channel+Spatial | $P_2$ | 92.20% | 0.9206 | 23,631,368 | 8.37 |
| Channel+Spatial | $P_3$ | 93.83% | 0.9370 | 23,730,216 | 7.70 |
| Channel+Spatial | $P_4$ | 93.65% | 0.9367 | 24,124,520 | 8.41 |
| Self-Attention | $P_3$ | 94.19% | 0.9410 | 24,908,678 | 8.20 |
| ECA | $P_3$ | **94.56%** | **0.9459** | 23,597,962 | 8.33 |
| CBAM | $P_3$ | 94.37% | 0.9444 | 23,730,216 | 8.39 |

None of the attention variants clearly beat the plain ResNet50. ECA scored highest (94.56%),
but its lead is a single extra correct image out of 551, so the differences between most models
are within what a re-run could change. Attention type mattered less than attention position:
$P_3$ (93.83%) and $P_4$ (93.65%) clearly outperformed the early positions $P_1$ (92.56%) and
$P_2$ (92.20%), while inserting attention at several stages ($P_3+P_4$, $P_2+P_3+P_4$) gave no
gain (93.47%) and multiplied the added parameters five-fold and peak GPU memory by about 2.6. The order of channel and spatial
attention made no difference (93.83% for both), a larger spatial kernel helped steadily
($3\times3$: 92.01%, $5\times5$: 93.10%, $7\times7$: 93.65%), and stacking channel, spatial and
self-attention was the worst model (91.65%). The lightweight ECA module (5 added parameters)
matched or beat the heavier SE, CBAM and self-attention modules, so more parameters did not
mean better accuracy. Qualitatively, the attention modules changed the features very little
(channel weights saturate near 1, spatial maps mostly respond to image borders, self-attention
forms a single small hotspot) and the Grad-CAM maps stayed close to the baseline's; the main
error for both the baseline and the attention model is roses predicted as tulips.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── Deep_Learning_Lab_8,9_Attention_ResNet50.ipynb   # full experiment, run top to bottom (GPU runtime)
├── report/
│   ├── main.tex                                     # LaTeX lab report code
│   └── figures/
│       └── *.pdf                                    # 600 dpi plot exports referenced by main.tex
└── plots/
    └── *.eps                                        # 600 dpi plots generated by the notebook
```

## Dataset

**TensorFlow Flowers Dataset** — downloaded once by the notebook through
`tf.keras.utils.get_file` from the TensorFlow example-images archive, so no manual download is
required.

- 3,670 RGB images in 5 classes: daisy, dandelion, roses, sunflowers, tulips
- Every image resized to $224\times224\times3$ and preprocessed with ResNet50's
  `preprocess_input`
- Augmentation (training set only, identical for every model): random horizontal flip, small
  rotation, random zoom, random translation and mild contrast variation
- Class-stratified 70/15/15 split with a fixed seed (2,569 training / 550 validation / 551 test
  images), reused for every experiment so that differences in performance come from the model
  and not from the split
- Labels are used for training and evaluation only; the test set is used once per model for the
  reported metrics

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set up a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook "Deep_Learning_Lab_8,9_Attention_ResNet50.ipynb"
```

or open it directly in [Google Colab](https://colab.research.google.com/) / Kaggle (GPU
runtime strongly recommended - the notebook trains 16 models, each for 20 epochs, which takes
roughly 5 hours on a Tesla T4) and run all cells top to bottom. On Kaggle, turn Internet on in the
notebook settings. The notebook downloads the dataset and the ImageNet weights by itself, and every trained model is cached under
`outputs/results` and `outputs/weights`, so an interrupted run resumes by simply re-running the
experiment cell - finished models are skipped. All figures are written to `outputs/figures`
(`png`, `pdf` and `eps` folders) together with one downloadable zip archive per format.

Figures use Times New Roman. On Linux (Colab/Kaggle) the notebook tries to install it, and falls
back to a similar serif font with a printed warning if that is not possible.

## Building the Report

The LaTeX report lives in `report/main.tex` and pulls in the figures from `report/figures/`.
The figures are already PDF (not EPS), so no `-shell-escape` flag is required.

1. Upload `report/main.tex` and everything in `report/figures/` into an
   [Overleaf](https://www.overleaf.com/) project (same folder level - keep the `figures/`
   subfolder as is).
2. Compile with pdfLaTeX.

Or locally:

```bash
cd report
pdflatex main.tex
pdflatex main.tex
```

## Author

**Madhav.S**
B.Tech Artificial Intelligence & Data Science, Section AIDS 'B'
Register Number: 24011101066
