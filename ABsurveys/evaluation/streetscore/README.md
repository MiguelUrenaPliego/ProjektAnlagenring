# Street Perception Scoring — Scientific & Technical Usage Guide

Fine-tune Vision Transformer (ViT) models on pairwise human preference comparison data and score collections of street-level images on perception metrics (e.g., safety, walkability, liveliness, beauty).

This codebase represents a complete end-to-end pipeline equipped with a **Multi-Task Differentiable Optimization engine** (using standard, pairwise, or mixed loss formulations), **focused custom metric monitoring**, and a **robust Dual Uncertainty framework** (Aleatoric Shannon Entropy and Epistemic MC-Dropout).

---

## 📂 Code Base Structure

```
.
├── main.py          # Entry point — configuration, hyperparameters, and dataset assembly
├── train.py         # Custom batch training loop, epoch state trackers, and metric plotting
├── inference.py     # Batch image scoring and dual-uncertainty calculations
├── model.py         # Network architecture (ViT-B/16 backbone + MLP Head)
├── losses.py        # Loss and accuracy functions (cross-entropy, pair, mixed loss)
├── uncertainty.py   # Information theory utilities and MC-Dropout wrappers
└── models/
    ├── default_models/          # Pre-trained HF checkpoints (automatically downloaded)
    │   └── safety.pth
    └── YourProject/             # Fine-tuned model directory (outputs: .pth, _history.csv, _curves.jpg)
```

---

## 🔬 Scientific & Technical Aspects

### 1. Differentiable Pairwise Optimization

Standard perception modeling often transforms pairwise rankings into artificial single-image pseudo-labels, which loses important relational ranking context. Our framework implements direct, differentiable pairwise ranking optimization:

* **Pair Loss (`pair_loss`)**:
  In `losses.py`, we optimize the model directly using a score difference formulation. 
  For strict comparisons, the loss is the difference between the rejected score and the preferred score:
  $$L_{\text{strict}} = s_{\text{rejected}} - s_{\text{preferred}}$$
  This makes the loss negative when the selected/preferred image score is higher than the non-selected image score—meaning the lower the loss score, the better the model's selection.
  
  For equal or tied comparison pairs (where humans selected that both images are equal/tied), we apply the absolute difference:
  $$L_{\text{tie}} = |s_A - s_B|$$
  This forces both scores to be equal (ideally 0 difference). Any difference is penalized, meaning lower values are better.
  
  The total pairwise loss is summed across all pairs and normalized by dividing by the total number of images of the set:
  $$L_{\text{pair}} = \frac{1}{2 \cdot N} \sum_{i=1}^N L_i$$

* **Pair Accuracy / Score (`pair_accuracy`)**:
  Calculates a continuous performance score over the comparison pairs.
  For strict comparisons, the score contribution is:
  $$\text{Score}_{\text{strict}} = s_{\text{preferred}} - s_{\text{rejected}}$$
  For equal/tie comparisons, any difference is penalized:
  $$\text{Score}_{\text{tie}} = -|s_A - s_B|$$
  The contributions of all pairs are added and divided by the number of pairs:
  $$\text{Score}_{\text{pair}} = \frac{1}{N} \sum_{i=1}^N \text{Score}_i$$

* **Mixed Loss (`mixed_loss`)**:
  To ensure that the influence of each loss is exactly equal (50% contribution), regardless of scale differences (e.g., if pairwise loss magnitude is much higher or lower than cross-entropy loss), we dynamically normalize each term. 
  
  The pairwise ranking loss is scaled to match the detached absolute scale of the cross-entropy loss. To prevent loss cancellation (i.e., cross-entropy and a negative pair-loss canceling out to zero), we add an offset equal to the detached cross-entropy scale. This preserves 50/50 gradient flow while keeping the loss positive:
  $$L_{\text{mixed}} = 0.5 \cdot L_{\text{ce}} + 0.5 \cdot \left( L_{\text{pair}} \cdot \frac{|L_{\text{ce}}|_{\text{detached}}}{|L_{\text{pair}}|_{\text{detached}}} \right) + 0.5 \cdot |L_{\text{ce}}|_{\text{detached}}$$

---

### 2. Custom Hyperparameter Selection & Strict Metric Logging

In `main.py`, you configure how the model learns and what it optimizes:

```python
# Select optimization loss function: "pair", "crossentropy", or "mixed"
LOSS: str = "mixed"

# Track accuracy/performance and save checkpoints on improvement: "pair" or "single"
ACCURACY: str = "pair"
```

To prevent "AI Slop" and noisy outputs, **the system strictly respects these hyperparameters during training**:
* **F1 Score (`f1_score`)**: When monitoring "single" image classification (or cross-entropy), the system uses a robust binary **F1 Score** instead of simple accuracy to better handle target distribution imbalances.
* **Logs & CSV**: The console logs and the output history file (e.g., `walk_history.csv`) only calculate, print, and save the active `train_loss`, `train_acc`, `val_loss`, and `val_acc` belonging to the chosen `LOSS` and `ACCURACY` settings.
* **Checkpoint Tracking**: A boolean `checkpoint` column in the history CSV keeps track of exactly when a model checkpoint was saved on disk (`True` on epochs where validation accuracy increased, `False` otherwise).
* **Plotting Curves**: The output visual curves (`walk_curves.jpg`) dynamically adjust their labels and plot lines to render only the selected metrics, eliminating irrelevant or unconfigured curves. All curves are plotted starting gracefully from epoch 0.
* **Validation Statistics**: Validation sets record and track the mean validation score, the standard deviation, and the average entropy of predictions under the column `entropy`.

---

### 3. Loss-Aware Dataset Splitting (`dataset.py`)

All splits are always performed **at the image level** — each `img_id` is randomly assigned to exactly one of train, val, or test. However, what happens to pairs that *cross* the split boundary (one image in train, one in val) depends on the active `LOSS` setting.

#### Pair / Mixed mode (`LOSS = "pair"` or `"mixed"`)

The pairwise ranking loss requires **both** images of a pair to be present in the same forward pass. A cross-split pair would expose a val-split image to the training gradient, leaking information across splits. The pipeline therefore applies `resolve_pairs_for_val_split` and **drops any pair whose two images land in different splits**.

```
img_id 1 → train  |  img_id 2 → train  |  img_id 3 → val
pair a: 1 vs 2, img 1 wins  →  train gets pair a  (both in train)
pair b: 1 vs 3, img 3 wins  →  DROPPED            (cross-split)
```

#### Crossentropy / Single-image mode (`LOSS = "crossentropy"`)

Cross-entropy loss is computed **per image independently** — the model never sees a pair simultaneously. There is therefore no leakage risk from cross-split pairs. The pipeline instead applies `resolve_pairs_image_level`, which splits pairs at the image level: each image contributes its label row only to the split that owns that image.

```
img_id 1 → train  |  img_id 2 → train  |  img_id 3 → val
pair a: 1 vs 2, img 1 wins  →  train: img 1 label 1, img 2 label 0
pair b: 1 vs 3, img 3 wins  →  train: img 1 label 0       (one image)
                             →  val:   img 3 label 1       (other image)
```

Cross-split pairs therefore contribute to *both* splits simultaneously, each side receiving only its own image. This is what the pipeline calls a **shared pair**: the pair is not duplicated wholesale, but each image appears exactly once in its own split's dataset. This maximises the training signal available from every comparison collected, particularly valuable when survey datasets are small.

The `SingleImageDataset` (in `train.py`) implements this: it loads one image per sample and returns the same tensor for both the A and B slots expected by the training loop. Under cross-entropy, `targets = cat([label, 1-label])`, so both slots receive the true binary label pair — mathematically equivalent to standard binary cross-entropy on a single image, with no leakage.

##### Impact on dataset size

With 312 pairs and a 70/30 train/val image split, the real-world difference is visible in the logs:

| Mode | Train samples | Val samples | Cross-split pairs |
|------|--------------|-------------|-------------------|
| pair / mixed | 173 pairs × 2 = 346 | 16 pairs × 2 = 32 | 123 dropped |
| crossentropy | **501** single-image rows | **167** single-image rows | 123 shared |

The crossentropy mode recovers all 123 cross-split pairs, yielding ~45% more training signal from the same survey data.

---

### 4. Advanced Features & First-Release Details

#### A. Robust Resuming with Checkpoint Realignment & Validation Baseline Recomputation
When resuming training from existing models (e.g., loading an existing `walk.pth`), our framework employs a smart checkpoint alignment and validation recomputation utility:
* **History Truncation**: It automatically scans the existing `_history.csv` to find the **last saved epoch** (the highest epoch where `checkpoint` was marked `True`) and **truncates** any noisy or unsaved training records that occurred *after* that last saved checkpoint to align the CSV perfectly with the saved weights.
* **Active Validation Baseline Recomputation**: Regardless of the epoch we are starting or resuming from, the training engine **always recomputes the validation accuracy/score and loss** right before training begins. This is critical because your loss function (`LOSS`), monitoring parameters (`ACCURACY`), or performance metrics could have changed between runs. Recomputing the baseline guarantees that the active `best_accuracy` is perfectly aligned with your active setup, preventing misaligned checkpoint comparisons.
* **Checkpoint Tracking Recovery**: It recovers the newly recomputed baseline as the correct `best_accuracy` to guarantee that subsequent epochs only trigger model checkpoints on actual, verified performance improvements.

#### B. Street-Level Perspective Image Augmentations & Dataset Doubling
To maximize training generalization and robust street feature learning, our pipeline uses augmentations specifically designed for street-level imagery paired with an expanded training pipeline:
* **Aspect Ratio Preservation (`CenterCropToSquare`)**: To prevent spatial distortion (i.e. stretching or squeezing) when loading arbitrary-sized street images for training or inference, we apply a custom `CenterCropToSquare` pre-processing step. This module dynamically finds the shorter dimension of any input image, crops a perfect 1:1 square from its center, and only then resizes or augments the image to the model's required $(384, 384)$ resolution. This guarantees the model only ever processes undistorted scenes with correct perspective geometry.
* **Horizontal Flips (`RandomHorizontalFlip`)**: Applied with $50\%$ probability. Note that top-down vertical flips are strictly excluded, as vertical inversion violates perspective physics for streetscapes.
* **Contrast & Saturation Jitter (`ColorJitter`)**: Adjusts contrast and saturation by up to $20\%$ to make the model robust to varying weather, daylight, and camera exposure conditions.
* **Gentle Scale & Aspect Cropping (`RandomResizedCrop`)**: Applies a gentle scale of $90\%$ to $100\%$ and aspect ratios between $0.95$ and $1.05$. This prevents losing crucial peripheral objects (like sidewalks, trees, or lane markers) while still providing robust scale invariance.
* **Dataset Size Expansion**: Because of these new random augmentations, the training dataset length is **doubled** in memory (`double_length=True`). This allows the model to see multiple unique augmented views of the same images across training iterations without overfitting, dramatically improving representation generalization.
* **Equal Pair (`=`) Verification & Logging**: The dataset loader scans and prints the exact number of tie (`=`) comparison pairs. It explicitly duplicates each tie comparison into two symmetric training samples:
  1. Image A as preferred (label 1), with `is_tie = 1`.
  2. Image B as preferred (label 0), with `is_tie = 1`.
  This ensures balanced preference labels for ties, and the dataset prints a confirmation of this symmetric doubling during initialization.

---

### 5. Dual Uncertainty Framework (`uncertainty.py`)

Street images are inherently complex. To determine how sure the model is about its scores, we implement a dual uncertainty framework:

#### A. Aleatoric (Entropy-Based) Uncertainty (`entropy_{metric}`)
Aleatoric uncertainty represents the inherent statistical noise or ambiguity in the data itself (e.g., a street view that looks partially safe and partially unsafe). 

We measure this by computing the binary Shannon entropy of the predicted probability $p$:
$$H(p) = -p \log_2(p) - (1-p) \log_2(1-p)$$

##### ❓ Scientific Inquiry: "If the model outputs a score of 5.0, how sure is it?"
* **Answer**: A predicted score of **5.0** maps directly to a probability $p = 0.5$ of being preferred. At $p = 0.5$, Shannon entropy reaches its absolute **mathematical maximum of 1.0 bit**. 
* This indicates **complete neutrality or absolute ambiguity**—the model is completely unsure and indicates a perfect tie.
* For confident predictions, like a score of **9.5** ($p = 0.95$), the entropy drops to a low **0.286 bits**, translating to a confidence of **~71.4%**.
* In `scores.csv`, this value is logged directly as `entropy_{metric}` (e.g., `entropy_walk`).

#### B. Epistemic (MC-Dropout) Uncertainty (`uncertainty_mc_{metric}`)
Epistemic uncertainty represents model parameter uncertainty (i.e., whether the model is unfamiliar with a particular type of image due to lack of representation in the training set). 

* **The Scientific Challenge**: PyTorch's standard pre-trained Vision Transformers (`vit_b_16`) are loaded with a default dropout rate `p = 0.0`. Standard Monte Carlo Dropout passes on such models would yield an epistemic standard deviation of exactly `0.0` across all passes because no neurons are ever stochastic.
* **Our Solution**: In both `uncertainty.py` and `inference.py`, we implement a robust dynamic layer patcher. When executing stochastic MC-Dropout passes (`INFERENCE_MC_PASSES > 1`), any dropout layer encountered with `p == 0.0` is dynamically configured to a rate of `p = 0.1` before training mode is enforced. This ensures the model drops features stochastically, producing a true epistemic variance that measures model parameter consistency.

---

## 🚀 Quick Start

1. Define your data paths, metrics, and desired optimization criteria in the configuration block of `main.py`:
   ```python
   LOSS = "mixed"        # Options: "pair", "crossentropy", or "mixed"
   ACCURACY = "pair"    # Options: "pair" or "single"
   ```
2. Run the end-to-end pipeline:
   ```bash
   python main.py
   ```
   This script handles the complete dataset assembly, trains or resumes the Vision Transformer model, logs the exact progress, and writes final scored predictions alongside their dual uncertainties to `scores.csv` under the clean column headers `{metric}`, `entropy_{metric}`, and `uncertainty_mc_{metric}`.