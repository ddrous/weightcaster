#%%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
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
    "hidden_size": 32,
    "depth": 3,
    "learning_rate": 1e-3,
    "epochs": 3000,
    "batch_size": 64,
}

print(f"Config seed is: {CONFIG['seed']}")

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

# Load Data
data, segs = gen_data_air()
train_mask = (segs == 0)
test_mask = (segs == 1)

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
    model, opt_state, loss = make_step(model, opt_state, X_train, Y_train)
    
    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss:.6f}")

# --- 5. EVALUATION ---

@eqx.filter_jit
def predict(model, x):
    return jax.vmap(model)(x)

y_pred_train = predict(model, X_train)
y_pred_test = predict(model, X_test)

train_mse = jnp.mean((Y_train - y_pred_train)**2)
test_mse = jnp.mean((Y_test - y_pred_test)**2)

print("-" * 30)
print(f"Mean Train MSE: {train_mse:.4f}")
print(f"Mean Test MSE:  {test_mse:.4f}")
print("-" * 30)


#%%
# --- 6. PLOTTING ---
# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(10, 6))

# 1. Plot Truth (Blue for Train, Orange for Test)
# Matching the color scheme from your attached Engression code
ax.scatter(X_train, Y_train, c='blue', s=20, alpha=0.1, label="Train Data")
ax.scatter(X_test, Y_test, c='orange', s=20, alpha=0.1, label="Test Data")

def plot_nn_results(X, y_pred, color, label_prefix):
    # Sort for clean mean line plotting
    sort_indices = jnp.argsort(X.flatten())
    X_sorted = X.flatten()[sort_indices]
    y_sorted = y_pred.flatten()[sort_indices]

    # 1. Plot Mean Line (The learned function)
    # High alpha to be clearly visible
    ax.plot(X_sorted, y_sorted, c=color, linewidth=3, label=f"Mean {label_prefix}", alpha=0.8)
    
    # 2. Plot Prediction Density (Scatter with +)
    # Even though NN is deterministic, this visualizes where the model places density
    # Matching the 'alpha=0.4' and 'marker=+' from your Engression example
    ax.scatter(X_sorted, y_sorted, c=color, s=40, alpha=0.4, 
               linewidth=1, marker='+')

# 2. Plot Train Predictions (Green)
plot_nn_results(X_train, y_pred_train, color='green', label_prefix="Train Pred.")

# 3. Plot Test Predictions (Red)
plot_nn_results(X_test, y_pred_test, color='red', label_prefix="Test Pred.")

ax.set_title("Standard MLP", fontsize=26)
ax.set_ylim(-3, 5)
ax.grid(True, alpha=0.1)
plt.xlabel("O3")
plt.ylabel("NOx")
ax.legend(loc='upper right')

plt.tight_layout()
plt.draw()
plt.savefig("plots/standard_nn_predictions.pdf")
plt.show()