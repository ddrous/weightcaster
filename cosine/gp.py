#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel as C

# Set plotting style to match requested format
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['lines.markersize'] = 10

# --- 1. CONFIGURATION ---
# Using the final active values from your provided dictionary
CONFIG = {
    "seed": 2028,
    "data_samples": 2000,
    "noise_std": 0.005,
    "segments": 11,
    "x_range": [-1.5, 1.5],
    "train_seg_ids": [2, 3, 4, 5, 6, 7, 8], 
}

print(f"Config seed is: {CONFIG['seed']}")

# --- 2. DATA GENERATION ---
def gen_data(seed, n_samples, n_segments=3, x_range=[-3.0, 3.0], noise_std=0.1):
    """
    Generates classical 1D-1D regression dataset: y = cos(10x) + 0.5x
    """
    np.random.seed(seed)
    
    # 1. Generate X uniformly across the full range
    x_min, x_max = x_range
    X = np.random.uniform(x_min, x_max, n_samples)
    
    # 2. Define the unchanging relation P(Y|X)
    Y = np.cos(10 * X) + 0.5 * X
    
    # Add noise
    noise = np.random.normal(0, noise_std, n_samples)
    Y += noise
    
    # 3. Create Segments spatially
    bins = np.linspace(x_min, x_max, n_segments + 1)
    segs = np.digitize(X, bins) - 1
    segs = np.clip(segs, 0, n_segments - 1)
    
    # data shape: (N, 2) -> [x, y]
    data = np.column_stack((X, Y))
    return data, segs

# Generate Data
data, segs = gen_data(
    CONFIG["seed"], 
    CONFIG["data_samples"], 
    CONFIG["segments"], 
    CONFIG["x_range"], 
    CONFIG["noise_std"]
)

# Split into Train/Test based on segments
train_mask = np.isin(segs, CONFIG["train_seg_ids"])
test_mask = ~train_mask

X_train = data[train_mask, 0][:, None] # Shape (N, 1)
Y_train = data[train_mask, 1]          # Shape (N,)
X_test = data[test_mask, 0][:, None]
Y_test = data[test_mask, 1]

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# --- 3. GAUSSIAN PROCESS MODEL ---

# Kernel Definition:
# 1. C(1.0) * RBF: Captures the wiggles (cosine)
# 2. DotProduct: Captures the linear trend (0.5x) allowing extrapolation
# 3. WhiteKernel: Captures the noise_std
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e1)) + \
         DotProduct(sigma_0=0.0) + \
         WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-6, 1e-2))

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=CONFIG["seed"])

print("Fitting Gaussian Process...")
gp.fit(X_train, Y_train)
print(f"Learned kernel: {gp.kernel_}")

# Inference
# We compute predictions for both train and test sets to calculate losses and plot
y_pred_train, y_std_train = gp.predict(X_train, return_std=True)
y_pred_test, y_std_test = gp.predict(X_test, return_std=True)

# Calculate Mean Losses (MSE)
train_mse = np.mean((Y_train - y_pred_train)**2)
test_mse = np.mean((Y_test - y_pred_test)**2)

print("-" * 30)
print(f"Mean Train MSE: {train_mse:.6f}")
print(f"Mean Test MSE:  {test_mse:.6f}")
print("-" * 30)


#%%
# --- 4. PLOTTING ---

fig, ax = plt.subplots(figsize=(10, 6))

# 1. Plot True Data Line (Combined Train + Test sorted)
# We combine all data to draw the black ground truth line
X_full = np.concatenate([X_train, X_test])
Y_full = np.concatenate([Y_train, Y_test])
sort_ix = np.argsort(X_full.flatten())
ax.plot(X_full.flatten()[sort_ix], Y_full.flatten()[sort_ix], c='k', linewidth=4, alpha=1, label="True Func.")

# Helper function to replicate the exact "scatter + vlines" style
def plot_gp_results(X, y_pred, y_std, color_mean, color_band, label_prefix):
    # Sort for cleaner plotting logic (though scatter doesn't strictly need it, vlines logic is easier)
    sort_indices = np.argsort(X.flatten())
    X_sorted = X.flatten()[sort_indices]
    mu_sorted = y_pred.flatten()[sort_indices]
    sigma_sorted = y_std.flatten()[sort_indices]

    # Plot Mean (Scatter with '+')
    ax.scatter(X_sorted, mu_sorted, c=color_mean, s=50, alpha=0.6, 
               label=f"Mean {label_prefix}", linewidth=2, marker='+')

    # Plot Uncertainty (Vertical Lines at every point)
    # Using the multiplier 2 as in the original code (Mean +/- 2 Std)
    multiplier = 2.0
    
    # We create a collection of lines. We only label the first one to avoid legend clutter.
    # Note: iterating 2000 points in python for vlines is slow, so we use ax.vlines which is vectorized
    ax.vlines(X_sorted, 
              mu_sorted - multiplier * sigma_sorted, 
              mu_sorted + multiplier * sigma_sorted, 
              colors=color_band, alpha=0.08)
    
    # Fix legend duplication (matplotlib handles duplicate labels automatically, but explicit control is safer)
    # The plot above adds the label to the line collection.

# 2. Plot Train Predictions (Green)
plot_gp_results(X_train, y_pred_train, y_std_train, 
                color_mean='green', color_band='green', label_prefix="Train Pred.")

# 3. Plot Test Predictions (Red)
plot_gp_results(X_test, y_pred_test, y_std_test, 
                color_mean='red', color_band='red', label_prefix="Test Pred.")

# Formatting
ax.set_title("Gaussian Process", fontsize=26)
# ax.set_ylim(np.min(Y_train)-1, np.max(Y_train)+1)
ax.grid(True, alpha=0.1)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

ax.set_ylim(-2.5, 2.5)

# De-duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# Reordering to match original: True Data, Train Mean, Train Unc, Test Mean, Test Unc
order = ["True Func.", "Mean Train Pred.", "Train Pred. Uncertainty", "Mean Test Pred.", "Test Pred. Uncertainty"]
ordered_handles = [by_label[l] for l in order if l in by_label]
ordered_labels = [l for l in order if l in by_label]
ax.legend(ordered_handles, ordered_labels, loc='lower right')

plt.tight_layout()
# plt.show()

plt.draw()
plt.savefig("plots/gp_predictions_uncertainty.pdf")

# %%
