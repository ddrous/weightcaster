#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['lines.markersize'] = 10

# --- 1. CONFIGURATION ---
# Using the specific parameters from your Cosine Problem definition
CONFIG = {
    "seed": 2028,
    "data_samples": 2000,
    "noise_std": 0.005,
    "segments": 11,
    "x_range": [-1.5, 1.5],
    "train_seg_ids": [2, 3, 4, 5, 6, 7, 8],
    
    # Engression Hyperparameters
    "hidden_dim": 128,
    "num_layer": 3,       # Depth to capture the frequency
    "noise_dim": 50,
    "lr": 1e-3,
    "epochs": 2000,       # Sufficient epochs for convergence
    "batch_size": 128,
    "device": "cpu"       # Change to "cuda" if available
}

print(f"Config seed is: {CONFIG['seed']}")
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# --- 2. DATA GENERATION (Cosine Problem) ---

def gen_data(seed, n_samples, n_segments=3, x_range=[-3.0, 3.0], noise_std=0.1):
    """
    Generates the classical 1D regression dataset: y = cos(10x) + 0.5x
    """
    np.random.seed(seed)
    
    # 1. Generate X uniformly across the full range
    x_min, x_max = x_range
    X = np.random.uniform(x_min, x_max, n_samples)
    
    # 2. Define the relation P(Y|X)
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

# --- 3. UTILITIES ---

def vectorize(x, multichannel=False):
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    if len(x.shape) == 2:
        return x
    else:
        if not multichannel:
            return x.reshape(x.shape[0], -1)
        else:
            return x.reshape(x.shape[0], x.shape[1], -1)

def _compute_norm(tensor, p, dim):
    return torch.norm(tensor, p=p, dim=dim)

def make_dataloader(x, y=None, batch_size=32, shuffle=True):
    if y is None:
        dataset = TensorDataset(x)
    else:
        dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# --- 4. LOSS FUNCTION (Energy Score) ---

def energy_loss_two_sample(x0, x, xp, beta=1, verbose=True):
    EPS = 1e-5
    x0 = vectorize(x0)
    x = vectorize(x)
    xp = vectorize(xp)
    
    # Term 1: Attraction (Match Data)
    norm_x_x0 = _compute_norm(x - x0, 2, dim=1)
    term1_a = (norm_x_x0 + EPS).pow(beta).mean()
    
    norm_xp_x0 = _compute_norm(xp - x0, 2, dim=1)
    term1_b = (norm_xp_x0 + EPS).pow(beta).mean()
    
    s1 = 0.5 * (term1_a + term1_b)
    
    # Term 2: Repulsion (Diversity)
    norm_x_xp = _compute_norm(x - xp, 2, dim=1)
    s2 = (norm_x_xp + EPS).pow(beta).mean()
    
    loss = s1 - s2 / 2
    return loss

# --- 5. MODEL ARCHITECTURE ---

def get_act_func(name):
    if name == "relu": return nn.ReLU(inplace=True)
    elif name == "softplus": return nn.Softplus()
    else: return None

class StoLayer(nn.Module):    
    def __init__(self, in_dim, out_dim, noise_dim=100, add_bn=True, out_act="relu", noise_std=1.0):
        super().__init__()
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        
        layers = [nn.Linear(in_dim + noise_dim, out_dim)]
        if add_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        
        self.layer = nn.Sequential(*layers)
        self.out_act = get_act_func(out_act)
    
    def forward(self, x):
        device = x.device
        eps = torch.randn(x.size(0), self.noise_dim, device=device) * self.noise_std
        out = torch.cat([x, eps], dim=1)
        out = self.layer(out)
        if self.out_act is not None:
            out = self.out_act(out)
        return out

class StoNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_layer=2, hidden_dim=100, noise_dim=100):
        super().__init__()
        # Input Layer
        self.input_layer = StoLayer(in_dim, hidden_dim, noise_dim, add_bn=True, out_act="softplus")
        
        # Hidden Layers
        hidden_layers = []
        for _ in range(num_layer - 2):
            hidden_layers.append(StoLayer(hidden_dim, hidden_dim, noise_dim, add_bn=True, out_act="softplus"))
        self.inter_layer = nn.Sequential(*hidden_layers)
        
        # Output Layer
        self.out_layer = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        out = self.input_layer(x)
        out = self.inter_layer(out)
        out = self.out_layer(out)
        return out

    @torch.no_grad()
    def sample(self, x, sample_size=100):
        # x shape: (N, dim)
        x_rep = x.repeat_interleave(sample_size, dim=0)
        samples = self.forward(x_rep)
        samples = samples.view(x.size(0), sample_size, -1)
        return samples

# --- 6. ENGRESSOR WRAPPER ---

class Engressor:
    def __init__(self, in_dim, out_dim, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.model = StoNet(in_dim, out_dim, 
                            num_layer=config["num_layer"], 
                            hidden_dim=config["hidden_dim"], 
                            noise_dim=config["noise_dim"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])
        
        # Stats for standardization
        self.x_mean = None; self.x_std = None
        self.y_mean = None; self.y_std = None

    def _update_stats(self, x, y):
        self.x_mean = torch.mean(x, dim=0).to(self.device)
        self.x_std = torch.std(x, dim=0).to(self.device) + 1e-5
        self.y_mean = torch.mean(y, dim=0).to(self.device)
        self.y_std = torch.std(y, dim=0).to(self.device) + 1e-5

    def train(self, x, y):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        x = vectorize(x); y = vectorize(y)
        
        # Standardize
        self._update_stats(x, y)
        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        
        loader = make_dataloader(x, y, batch_size=self.config["batch_size"])
        self.model.train()
        
        print("Starting training...")
        for epoch in range(self.config["epochs"]):
            for bx, by in loader:
                self.optimizer.zero_grad()
                # Draw two samples for energy loss
                y1 = self.model(bx)
                y2 = self.model(bx)
                loss = energy_loss_two_sample(by, y1, y2)
                loss.backward()
                self.optimizer.step()

    def predict(self, x, sample_size=100):
        self.model.eval()
        x_t = torch.tensor(x, dtype=torch.float32).to(self.device)
        x_t = vectorize(x_t)
        
        # Standardize Input
        x_t = (x_t - self.x_mean) / self.x_std
        
        # Sample
        samples = self.model.sample(x_t, sample_size=sample_size) # (N, samples, 1)
        
        # Unstandardize Output
        samples = samples * self.y_std + self.y_mean
        samples = samples.cpu().numpy().squeeze(-1)
        
        return samples.mean(axis=1), samples.std(axis=1)

# --- 7. MAIN EXECUTION ---

# 1. Generate Cosine Data
data, segs = gen_data(CONFIG["seed"], CONFIG["data_samples"], CONFIG["segments"], CONFIG["x_range"], CONFIG["noise_std"])

# 2. Split (Train on middle segments)
train_mask = np.isin(segs, CONFIG["train_seg_ids"])
test_mask = ~train_mask

X_train = data[train_mask, 0]
Y_train = data[train_mask, 1]
X_test = data[test_mask, 0]
Y_test = data[test_mask, 1]

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# 3. Train Engressor
model = Engressor(in_dim=1, out_dim=1, config=CONFIG)
model.train(X_train, Y_train)

# 4. Inference
print("Running Inference...")
train_mu, train_std = model.predict(X_train, sample_size=500)
test_mu, test_std = model.predict(X_test, sample_size=500)

# 5. Calculate MSE
train_mse = np.mean((Y_train - train_mu)**2)
test_mse = np.mean((Y_test - test_mu)**2)

print("-" * 30)
print(f"Mean Train MSE: {train_mse:.6f}")
print(f"Mean Test MSE:  {test_mse:.6f}")
print("-" * 30)

# --- 8. PLOTTING ---

#%%
fig, ax = plt.subplots(figsize=(10, 6))

# Plot True Data (Black Line for continuous ground truth)
X_full = np.concatenate([X_train, X_test])
Y_full = np.concatenate([Y_train, Y_test])
sort_ix = np.argsort(X_full)
ax.plot(X_full[sort_ix], Y_full[sort_ix], c='k', linewidth=4, alpha=1, label="True Func.")

def plot_engression_bands(X, mu, sigma, color_mean, color_band, label):
    # Sort for visual consistency
    sort_idx = np.argsort(X)
    X_s = X[sort_idx]
    mu_s = mu[sort_idx]
    sigma_s = sigma[sort_idx]
    
    # Plot Mean (Scatter with + marker)
    ax.scatter(X_s, mu_s, c=color_mean, s=50, alpha=0.6, 
               label=f"Mean {label}", linewidth=2, marker='+')
    
    # Plot Uncertainty (Vertical Lines)
    # Using 2 * sigma for approx 95% CI if Gaussian-like
    multiplier = 2.0
    ax.vlines(X_s, 
              mu_s - multiplier * sigma_s, 
              mu_s + multiplier * sigma_s, 
              colors=color_band, alpha=0.08)

# Plot Train (Green)
plot_engression_bands(X_train, train_mu, train_std, 'green', 'green', "Train Pred.")

# Plot Test (Red)
plot_engression_bands(X_test, test_mu, test_std, 'red', 'red', "Test Pred.")

ax.set_ylim(-2.5, 2.5)

ax.set_title("Engression", fontsize=26)
# ax.set_ylim(np.min(Y_train)-1, np.max(Y_train)+1)
ax.grid(True, alpha=0.1)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

# De-duplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
order = ["True Func.", "Mean Train Pred.", "Train Pred. Uncertainty", "Mean Test Pred.", "Test Pred. Uncertainty"]
ordered_handles = [by_label[l] for l in order if l in by_label]
ordered_labels = [l for l in order if l in by_label]
ax.legend(ordered_handles, ordered_labels, loc='lower right')

plt.tight_layout()
plt.show()