# Frankfurt Anlagenring Street Perception Map

An end-to-end multi-model, multi-metric street perception mapping and comparison system. It ingests survey responses and machine-learning predictions, aligns their distributions, resolves geographic positions, and compiles them into a beautiful, highly interactive Leaflet dashboard overlay.

---

## 📊 Score Normalization, Mean Alignment, and Scaling Math

To make two completely different score distributions (e.g., TrueSkill model scores vs. StreetScore deep learning predictions) directly comparable side-by-side without changing their relative rankings or ratios, the system applies a dual-distribution alignment algorithm located in `utils.py`. 

Here is the exact mathematical step-by-step process used to normalize, align, and scale the scores and their corresponding uncertainties:

### 1. Zero-Centering (Mean Alignment)
First, the mean ($\mu$) of each score distribution (TrueSkill $X$ and StreetScore $Y$) is calculated, and subtracted from each data point to align their centers perfectly at $0.0$:
$$x_{\text{zero}} = x - \mu_x$$
$$y_{\text{zero}} = y - \mu_y$$

### 2. Standardization (Unit Variance Scaling)
The standard deviations ($\sigma$) of the zero-centered series are computed:
$$\sigma_x = \text{std}(x_{\text{zero}})$$
$$\sigma_y = \text{std}(y_{\text{zero}})$$

We then standardize both series to unit variance (mean = $0.0$, standard deviation = $1.0$):
$$x_{\text{std}} = \frac{x_{\text{zero}}}{\sigma_x}$$
$$y_{\text{std}} = \frac{y_{\text{zero}}}{\sigma_y}$$

### 3. Absolute Bounds Optimization
To map both distributions to a global absolute scale of $[0.0, 10.0]$ without any hard clipping, we find the maximum absolute deviation from the mean across *both* distributions combined:
$$D_{\text{max}} = \max\left( \max(|x_{\text{std}}|), \max(|y_{\text{std}}|) \right)$$

This combined maximum value represents the single furthest point from the mean in either distribution.

### 4. Target Scale Calculation
We calculate a target standard deviation ($\sigma_{\text{target}}$) that will stretch or compress both distributions so that this maximum deviation is exactly $5.0$ (allowing the entire dataset to span perfectly within $[0.0, 10.0]$ when centered at $5.0$):
$$\sigma_{\text{target}} = \frac{5.0}{D_{\text{max}}}$$

### 5. Final Projection & Translation
We project the standardized scores back using the computed target standard deviation, and translate the center to $5.0$:
$$x_{\text{aligned}} = 5.0 + \left(x_{\text{std}} \cdot \sigma_{\text{target}}\right)$$
$$y_{\text{aligned}} = 5.0 + \left(y_{\text{std}} \cdot \sigma_{\text{target}}\right)$$

This mathematical transform guarantees that:
* Both models' means are perfectly aligned at **5.0**.
* Both models share the exact same standard deviation, matching their distributions.
* The absolute extreme data point in either dataset sits exactly at **0.0** or **10.0**, maximizing color contrast on the map without any value clipping.

### 6. Uncertainty Scaling Multipliers
Because the scores are scaled by the factors $M_x = \frac{\sigma_{\text{target}}}{\sigma_x}$ and $M_y = \frac{\sigma_{\text{target}}}{\sigma_y}$, their corresponding uncertainty values (standard error or entropy) must be scaled by the **exact same multipliers** to keep the comparison brackets proportionally accurate:
$$\text{Uncertainty}_x^{\text{scaled}} = \text{Uncertainty}_x \cdot M_x$$
$$\text{Uncertainty}_y^{\text{scaled}} = \text{Uncertainty}_y \cdot M_y$$

This ensures that the double-decker comparisons, error bars, and uncertainty maps show true relative confidence levels.

---

## 🛠️ Data Configuration & Flexibility

The pipeline (`main.py`) supports highly flexible configuration parameters:

### 1. Omission of Models (`None` handling)
If you do not wish to display one of the models (e.g. `streetscore` or `trueskill`) on the map:
* Set `STREETSCORE_DF_PATHS = None` or `TRUESKILL_DF_PATHS = None` in the `CONFIGURATION` section of `main.py`.
* The dashboard dynamically detects which models are missing, hides the model switch overlay, disables the "difference" comparison mode, and adjusts the layout beautifully.

### 2. Multi-CSV Concatenation
When your datasets are split across multiple files, you can define them as lists:
```python
TRUESKILL_DF_PATHS = [
    "/path/to/trueskill_walk_scores.csv",
    "/path/to/trueskill_bike_scores.csv"
]
```
The pipeline automatically loads and concatenates them into a single dataframe using `pd.concat`.

### 3. File Path Relativity & Integrity
* **ROOT_PATH**: All configuration paths in `main.py` are resolved relative to `ROOT_PATH`.
* **images.csv Relativity**: File paths declared inside `images.csv` are resolved relative to `images.csv`'s parent directory, making it highly portable.
* **Metadata Alignment**: The pipeline ignores stale path and type columns in `scores.csv` or model outputs. It merges coordinate, type, and scenario metadata using `img_df` as the single source of truth, preventing missing or incorrectly classified scores (e.g., ensuring bike/stay images display correctly).

---

## 🎛️ Interactive UI Filter Splits

If your dataset contains a `split` column (representing dataset partitions like train/test/val, or other categories), the dashboard will:
1. Show the split name in a dedicated badge inside the image popup details panel.
2. Automatically build a checkbox group under the control panel titled **"filter splits"**.
3. Allow you to activate or deactivate multiple split checkboxes to filter which markers show on the map in real-time.
4. Render an **"Other"** checkbox to filter any images that lack a split value.
5. If the `split` column is completely absent, the filter panel remains hidden, ensuring zero UI clutter.
