#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['lines.markersize'] = 10

# --- 1. CONFIGURATION ---
CONFIG = {
    "seed": 2028,
    "data_samples": 2000,
    "noise_std": 0.005,
    "segments": 11,
    "x_range": [-1.5, 1.5],
    "train_seg_ids": [2, 3, 4, 5, 6, 7, 8],
    
    # MLP Hyperparameters
    "hidden_size": 64,
    "depth": 3,
    "learning_rate": 1e-3,
    "epochs": 3000,
    "batch_size": 64
}

print(f"Config seed is: {CONFIG['seed']}")

# --- 2. DATA GENERATION ---
def gen_data(seed, n_samples, n_segments=3, x_range=[-3.0, 3.0], noise_std=0.1):
    np.random.seed(seed)
    
    # 1. Generate X uniformly
    x_min, x_max = x_range
    X = np.random.uniform(x_min, x_max, n_samples)
    
    # 2. Define P(Y|X): y = cos(10x) + 0.5x
    Y = np.cos(10 * X) + 0.5 * X
    
    # Add noise
    noise = np.random.normal(0, noise_std, n_samples)
    Y += noise
    
    # 3. Create Segments
    bins = np.linspace(x_min, x_max, n_segments + 1)
    segs = np.digitize(X, bins) - 1
    segs = np.clip(segs, 0, n_segments - 1)
    
    # Format
    data = np.column_stack((X, Y))
    return data, segs

# Generate Data
data, segs = gen_data(CONFIG["seed"], CONFIG["data_samples"], CONFIG["segments"], CONFIG["x_range"], CONFIG["noise_std"])

# Split
train_mask = np.isin(segs, CONFIG["train_seg_ids"])
test_mask = ~train_mask

X_train = jnp.array(data[train_mask, 0])[:, None]
Y_train = jnp.array(data[train_mask, 1])[:, None]
X_test = jnp.array(data[test_mask, 0])[:, None]
Y_test = jnp.array(data[test_mask, 1])[:, None]

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# --- 3. MODEL DEFINITION ---

class StandardMLP(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        # Simple MLP: Linear -> Tanh -> Linear -> Tanh -> Linear
        self.layers = [
            eqx.nn.Linear(1, CONFIG["hidden_size"], key=key1),
            jax.nn.tanh,
            eqx.nn.Linear(CONFIG["hidden_size"], CONFIG["hidden_size"], key=key2),
            jax.nn.tanh,
            eqx.nn.Linear(CONFIG["hidden_size"], CONFIG["hidden_size"], key=key3),
            jax.nn.tanh,
            eqx.nn.Linear(CONFIG["hidden_size"], 1, key=key4)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Initialize
key = jax.random.PRNGKey(CONFIG["seed"])
model = StandardMLP(key)
opt = optax.adam(CONFIG["learning_rate"])
opt_state = opt.init(eqx.filter(model, eqx.is_array))

# --- 4. TRAINING LOOP ---

@eqx.filter_value_and_grad
def compute_loss(model, x, y):
    pred = jax.vmap(model)(x)
    return jnp.mean((pred - y) ** 2)

@eqx.filter_jit
def make_step(model, opt_state, x, y):
    loss, grads = compute_loss(model, x, y)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

print("Starting training...")
for epoch in range(CONFIG["epochs"]):
    # Full batch gradient descent for simplicity
    model, opt_state, loss = make_step(model, opt_state, X_train, Y_train)
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f}")

# --- 5. EVALUATION ---

@eqx.filter_jit
def predict(model, x):
    return jax.vmap(model)(x)

y_pred_train = predict(model, X_train)
y_pred_test = predict(model, X_test)

train_mse = jnp.mean((Y_train - y_pred_train)**2)
test_mse = jnp.mean((Y_test - y_pred_test)**2)

print("-" * 30)
print(f"Mean Train MSE: {train_mse:.6f}")
print(f"Mean Test MSE:  {test_mse:.6f}")
print("-" * 30)

#%%
# --- 6. PLOTTING ---

fig, ax = plt.subplots(figsize=(10, 6))

# 1. Plot True Data Line (Combined Train + Test sorted)
X_full = jnp.concatenate([X_train, X_test])
Y_full = jnp.concatenate([Y_train, Y_test])
sort_ix = jnp.argsort(X_full.flatten())
ax.plot(X_full.flatten()[sort_ix], Y_full.flatten()[sort_ix], c='k', linewidth=4, alpha=1, label="True Func.")

# Helper to plot markers
def plot_results(X, y_pred, color, label):
    # Sort for cleaner plotting logic (optional for scatter, but good practice)
    sort_indices = jnp.argsort(X.flatten())
    X_sorted = X.flatten()[sort_indices]
    y_sorted = y_pred.flatten()[sort_indices]

    # Plot Point Predictions (Scatter with '+')
    ax.scatter(X_sorted, y_sorted, c=color, s=50, alpha=0.6, 
               label=label, linewidth=2, marker='+')

# 2. Plot Train Predictions (Green)
plot_results(X_train, y_pred_train, color='green', label="Train Pred.")

# 3. Plot Test Predictions (Red)
plot_results(X_test, y_pred_test, color='red', label="Test Pred.")

ax.set_ylim(-2.5, 2.5)

# Formatting
ax.set_title("Standard MLP", fontsize=26)
# ax.set_ylim(jnp.min(Y_train)-1, jnp.max(Y_train)+1)
ax.grid(True, alpha=0.1)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

# Legend
ax.legend(loc='lower right')

plt.tight_layout()
plt.draw()
plt.savefig("plots/nn_predictions.pdf")
