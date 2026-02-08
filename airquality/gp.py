#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import os

# Set plotting style
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['lines.markersize'] = 10

# --- 1. CONFIGURATION ---
CONFIG = {
    "seed": 2028,
    "max_samples": 1000,  # CRITICAL: Limit training data size for GP speed
}

print(f"Config seed is: {CONFIG['seed']}")
np.random.seed(CONFIG["seed"])

# --- 2. DATA GENERATION ---
def gen_data_air():
    data_path = "air_quality.csv"
    
    if not os.path.exists(data_path):
        print("Warning: 'air_quality.csv' not found. Using synthetic proxy.")
        np.random.seed(42); X = np.random.uniform(-2, 3, 2000); Y = np.sin(X) + np.random.normal(0, 0.1, 2000)
        segs = (X > 1.0).astype(int)
        return np.column_stack((X, Y)), segs

    df = pd.read_csv(data_path)
    try:
        if 'PT08.S3.NOx.' not in df.columns: df = pd.read_csv(data_path, sep=';', decimal=',')
    except: pass
    
    rename_map = {'PT08.S3(NOx)': 'PT08.S3.NOx.', 'PT08.S5(O3)': 'PT08.S5.O3.'}
    df.rename(columns=rename_map, inplace=True)
    df = df.dropna(subset=['PT08.S3.NOx.', 'PT08.S5.O3.'])

    scaler = StandardScaler()
    df[['PT08.S3.NOx.', 'PT08.S5.O3.']] = scaler.fit_transform(df[['PT08.S3.NOx.', 'PT08.S5.O3.']])

    X = df['PT08.S5.O3.'].values
    Y = df['PT08.S3.NOx.'].values
    segs = (X > 1.0).astype(int)
    data = np.column_stack((X, Y))
    return data, segs

# Load Data
data, segs = gen_data_air()
train_mask = (segs == 0)
test_mask = (segs == 1)

X_train_full = data[train_mask, 0][:, None]
Y_train_full = data[train_mask, 1]
X_test = data[test_mask, 0][:, None]
Y_test = data[test_mask, 1]

# --- SUBSAMPLING FOR SPEED ---
# GP scales cubicly O(N^3). We subsample training data to make it tractable.
if len(X_train_full) > CONFIG["max_samples"]:
    print(f"Subsampling training data from {len(X_train_full)} to {CONFIG['max_samples']} points for GP speed...")
    idx = np.random.choice(len(X_train_full), CONFIG['max_samples'], replace=False)
    X_train_gp = X_train_full[idx]
    Y_train_gp = Y_train_full[idx]
else:
    X_train_gp = X_train_full
    Y_train_gp = Y_train_full

print(f"GP Training samples: {len(X_train_gp)} | Test samples: {len(X_test)}")

# --- 3. GAUSSIAN PROCESS MODEL ---

# Kernel: 
# RBF: Captures local non-linearity
# DotProduct: Captures global trend (extrapolation)
# WhiteKernel: Captures noise
kernel = C(1.0) * RBF(length_scale=1.0) + \
         DotProduct(sigma_0=1.0) + \
         WhiteKernel(noise_level=0.1)

# n_restarts_optimizer=0 speeds it up significantly (runs optimization once)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=CONFIG["seed"])

print("Fitting Gaussian Process (this may still take a few seconds)...")
gp.fit(X_train_gp, Y_train_gp)
print(f"Learned kernel: {gp.kernel_}")

# --- 4. INFERENCE & PLOTTING ---

# Predict Mean and Std
# We predict on the FULL sets for evaluation, which is O(N_test * N_train), faster than training
train_mu, train_std = gp.predict(X_train_full, return_std=True)
test_mu, test_std = gp.predict(X_test, return_std=True)

# Calculate MSE
train_mse = np.mean((Y_train_full - train_mu)**2)
test_mse = np.mean((Y_test - test_mu)**2)
print("-" * 30)
print(f"Mean Train MSE: {train_mse:.4f}")
print(f"Mean Test MSE:  {test_mse:.4f}")
print("-" * 30)

#%%
# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(10, 6))

# 1. Plot Truth (Blue for Train, Orange for Test)
ax.scatter(X_train_full, Y_train_full, c='blue', s=20, alpha=0.1, label="Train Data")
ax.scatter(X_test, Y_test, c='orange', s=20, alpha=0.1, label="Test Data")

def plot_gp_results(X, mu, sigma, color, label_prefix):
    # Sort for clean mean line
    sort_idx = np.argsort(X.flatten())
    X_s = X.flatten()[sort_idx]
    mu_s = mu.flatten()[sort_idx]
    sigma_s = sigma.flatten()[sort_idx]
    
    # 1. Plot Mean Line
    ax.plot(X_s, mu_s, c=color, linewidth=3, label=f"Mean {label_prefix}", alpha=0.8)
    
    # 2. Plot Stochastic Cloud (Samples with +)
    # Generate 1 sample per point: y ~ N(mu, sigma)
    noise = np.random.normal(0, 1, size=mu_s.shape)
    y_sampled = mu_s + sigma_s * noise
    
    # Use simple scatter for the cloud, matched density
    ax.scatter(X_s, y_sampled, c=color, s=40, alpha=0.2, 
               linewidth=1, marker='+')

# 2. Plot Predictions (Green for Train, Red for Test)
plot_gp_results(X_train_full, train_mu, train_std, 'green', "Train Pred.")
plot_gp_results(X_test, test_mu, test_std, 'red', "Test Pred.")

ax.set_title("Gaussian Process", fontsize=26)
ax.set_ylim(-3, 5)
ax.grid(True, alpha=0.1)
plt.xlabel("O3")
plt.ylabel("NOx")
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig("plots/gp_predictions.pdf")
plt.show()