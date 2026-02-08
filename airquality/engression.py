#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    "hidden_dim": 128,    
    "num_layer": 3,       
    "noise_dim": 50,      
    "lr": 1e-3,
    "epochs": 2000,       
    "batch_size": 256,
    "device": "cpu"       
}

print(f"Config seed is: {CONFIG['seed']}")
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# --- 2. DATA GENERATION ---

def gen_data_air():
    data_path = "air_quality.csv" 
    
    if not os.path.exists(data_path):
        print("Warning: 'air_quality.csv' not found. Using synthetic proxy.")
        np.random.seed(42); X = np.random.uniform(-2, 3, 1000); Y = np.sin(X) + np.random.normal(0, 0.1, 1000)
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

# --- 3. UTILITIES ---

def vectorize(x, multichannel=False):
    if len(x.shape) == 1: return x.unsqueeze(1)
    if len(x.shape) == 2: return x
    else: return x.reshape(x.shape[0], -1)

def _compute_norm(tensor, p, dim):
    return torch.norm(tensor, p=p, dim=dim)

def make_dataloader(x, y=None, batch_size=32, shuffle=True):
    dataset = TensorDataset(x) if y is None else TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# --- 4. LOSS FUNCTION ---

def energy_loss_two_sample(x0, x, xp, beta=1):
    EPS = 1e-5
    x0 = vectorize(x0); x = vectorize(x); xp = vectorize(xp)
    
    norm_x_x0 = _compute_norm(x - x0, 2, dim=1)
    term1_a = (norm_x_x0 + EPS).pow(beta).mean()
    
    norm_xp_x0 = _compute_norm(xp - x0, 2, dim=1)
    term1_b = (norm_xp_x0 + EPS).pow(beta).mean()
    
    s1 = 0.5 * (term1_a + term1_b)
    
    norm_x_xp = _compute_norm(x - xp, 2, dim=1)
    s2 = (norm_x_xp + EPS).pow(beta).mean()
    
    return s1 - s2 / 2

# --- 5. MODEL ARCHITECTURE ---

class StoLayer(nn.Module):    
    def __init__(self, in_dim, out_dim, noise_dim=100, add_bn=True, out_act="relu", noise_std=1.0):
        super().__init__()
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        layers = [nn.Linear(in_dim + noise_dim, out_dim)]
        if add_bn: layers.append(nn.BatchNorm1d(out_dim))
        self.layer = nn.Sequential(*layers)
        self.out_act = nn.ReLU() if out_act=="relu" else nn.Softplus()
    
    def forward(self, x):
        device = x.device
        eps = torch.randn(x.size(0), self.noise_dim, device=device) * self.noise_std
        out = torch.cat([x, eps], dim=1)
        out = self.layer(out)
        return self.out_act(out)

class StoNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_layer=2, hidden_dim=100, noise_dim=100):
        super().__init__()
        self.input_layer = StoLayer(in_dim, hidden_dim, noise_dim, add_bn=True, out_act="softplus")
        hidden_layers = []
        for _ in range(num_layer - 2):
            hidden_layers.append(StoLayer(hidden_dim, hidden_dim, noise_dim, add_bn=True, out_act="softplus"))
        self.inter_layer = nn.Sequential(*hidden_layers)
        self.out_layer = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        return self.out_layer(self.inter_layer(self.input_layer(x)))

    @torch.no_grad()
    def sample(self, x, sample_size=100):
        x_rep = x.repeat_interleave(sample_size, dim=0)
        samples = self.forward(x_rep)
        return samples.view(x.size(0), sample_size, -1)

# --- 6. ENGRESSOR WRAPPER ---

class Engressor:
    def __init__(self, in_dim, out_dim, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.model = StoNet(in_dim, out_dim, config["num_layer"], config["hidden_dim"], config["noise_dim"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])
        
    def train(self, x, y):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        x = vectorize(x); y = vectorize(y)
        
        loader = make_dataloader(x, y, batch_size=self.config["batch_size"])
        self.model.train()
        
        for epoch in range(self.config["epochs"]):
            for bx, by in loader:
                self.optimizer.zero_grad()
                y1 = self.model(bx); y2 = self.model(bx)
                loss = energy_loss_two_sample(by, y1, y2)
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % 500 == 0: print(f"Epoch {epoch+1} complete")

    def predict(self, x, sample_size=100):
        self.model.eval()
        x_t = torch.tensor(x, dtype=torch.float32).to(self.device)
        x_t = vectorize(x_t)
        samples = self.model.sample(x_t, sample_size=sample_size) 
        samples = samples.cpu().numpy().squeeze(-1) # (N, sample_size)
        return samples

# --- 7. MAIN EXECUTION ---

# Load Data
data, segs = gen_data_air()
train_mask = (segs == 0); test_mask = (segs == 1)
X_train = data[train_mask, 0]; Y_train = data[train_mask, 1]
X_test = data[test_mask, 0]; Y_test = data[test_mask, 1]

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Train
model = Engressor(in_dim=1, out_dim=1, config=CONFIG)
model.train(X_train, Y_train)

# Inference (Get raw samples for cloud)
print("Running Inference...")
samples_train = model.predict(X_train, sample_size=50) # Fewer samples for cleaner plot
samples_test = model.predict(X_test, sample_size=50)

# Calculate means for line plot
train_mu = samples_train.mean(axis=1)
test_mu = samples_test.mean(axis=1)

# Calc MSE
train_mse = np.mean((Y_train - train_mu)**2)
test_mse = np.mean((Y_test - test_mu)**2)
print(f"Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")

#%%
# --- 8. PLOTTING ---
# --- 8. INFERENCE & PLOTTING (Fixed Density) ---

fig, ax = plt.subplots(figsize=(10, 6))

# 1. Plot Truth (Blue for Train, Orange for Test)
ax.scatter(X_train, Y_train, c='blue', s=20, alpha=0.1, label="Train Data")
ax.scatter(X_test, Y_test, c='orange', s=20, alpha=0.1, label="Test Data")

# Helper to calculate Mean vs Single Sample
def get_plot_data(x_data):
    # 1. Get stable Mean for the line plot (average of 100 samples)
    samples_many = model.predict(x_data, sample_size=100)
    mu = samples_many.mean(axis=1)
    
    # 2. Get single sample for the cloud plot (1 sample per point)
    # This ensures density matches the ground truth
    samples_single = model.predict(x_data, sample_size=1).flatten()
    
    return mu, samples_single

# Generate Data
train_mu, train_single = get_plot_data(X_train)
test_mu, test_single = get_plot_data(X_test)

def plot_engression_results(X, mu, single_samples, color, label_prefix):
    # Sort for clean mean line plotting
    sort_idx = np.argsort(X)
    X_s = X[sort_idx]
    mu_s = mu[sort_idx]
    
    # 1. Plot Mean Line
    ax.plot(X_s, mu_s, c=color, linewidth=3, label=f"Mean {label_prefix}", alpha=0.8)
    
    # 2. Plot Stochastic Cloud (1 prediction per 1 input point)
    ax.scatter(X, single_samples, c=color, s=40, alpha=0.4, 
               linewidth=1, marker='+')

# 2. Plot Predictions (Green for Train, Red for Test)
plot_engression_results(X_train, train_mu, train_single, 'green', "Train Pred.")
plot_engression_results(X_test, test_mu, test_single, 'red', "Test Pred.")

ax.set_title("Engression", fontsize=26)
ax.set_ylim(-3, 5)
ax.grid(True, alpha=0.1)
plt.xlabel("O3")
plt.ylabel("NOx")
ax.legend(loc='upper right')

plt.tight_layout()
plt.draw()
plt.savefig("plots/engression_predictions.pdf")
plt.show()