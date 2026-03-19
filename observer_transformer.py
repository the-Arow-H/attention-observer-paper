# %% [markdown]
# # Observer-Transformer Full Experiment Notebook
# "Attention + Observer is All You Need" – Hypothesis Test

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------- 1. Tokenizer & Data -------------------
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
print(f"Dataset length: {len(text):,} characters")

tokenizer = CharTokenizer(text)
print(f"Vocab size: {tokenizer.vocab_size}")

block_size = 128
batch_size = 32

def get_batch(split: str = "train"):
    # Simple random batch (no train/val split for speed – add one if you want)
    ix = torch.randint(len(text) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(tokenizer.encode(text[i:i+block_size])) for i in ix])
    y = torch.stack([torch.tensor(tokenizer.encode(text[i+1:i+block_size+1])) for i in ix])
    return x.to(device), y.to(device)

# ------------------- 2. ObserverTransformer (full class from earlier) -------------------
# PASTE THE ENTIRE ObserverTransformer CLASS HERE (the one I gave you previously)
# For brevity in this message I omit it – just copy-paste it from the previous response.
# (It is ~120 lines and unchanged.)

# ------------------- 3. Training Function -------------------
def train(model, epochs=5, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for _ in tqdm(range(200), desc=f"Epoch {epoch+1}"):   # 200 steps per epoch
            x, y = get_batch()
            optimizer.zero_grad()
            logits, _ = model(x, None, y) if hasattr(model, 'with_observer') else model(x)
            loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / 200
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")
    return losses

# ------------------- 4. Ablation Runner -------------------
def run_ablation():
    results = {}
    for obs in [True, False]:
        print(f"\n=== Running with observer = {obs} ===")
        model = ObserverTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=128,
            nhead=4,
            num_layers=3,
            observer_dim=32
        ).to(device)
        model.with_observer = obs
        
        losses = train(model, epochs=3)
        results[f"observer_{obs}"] = losses[-1]
        
        # Self-awareness probe
        probe_with, probe_without = run_self_modeling_probe(model, tokenizer, device)
        results[f"probe_{obs}"] = probe_with if obs else probe_without
        
    print("\n=== PAPER-READY RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    return results

# ------------------- 5. Self-Awareness Probe (from earlier) -------------------
# PASTE THE ENTIRE run_self_modeling_probe FUNCTION HERE (the 12-item probe you already have)

# ------------------- 6. Generation with Persistent Observer -------------------
@torch.no_grad()
def generate(prompt: str, max_new=200, temp=0.8, use_observer=True):
    model.eval()
    model.with_observer = use_observer
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    obs_state = None
    print("Prompt:", prompt, end="")
    
    for _ in range(max_new):
        logits, obs_state = model(x, obs_state)
        logits = logits[:, -1] / temp
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        x = torch.cat([x, next_id], dim=1)
        print(tokenizer.decode([next_id.item()]), end="", flush=True)
    print("\n")

# ------------------- 7. Observer Gate Collapse Visualization -------------------
@torch.no_grad()
def plot_gate_collapse():
    model = ObserverTransformer(...).to(device)  # load your trained model
    model.with_observer = True
    gates = []
    for _ in range(50):
        x, _ = get_batch()
        _, obs_state = model(x, None)
        gate = model.gate_net(obs_state).mean().item()
        gates.append(gate)
    plt.plot(gates)
    plt.title("Observer Gate Strength (→ 1.0 = collapse/measurement)")
    plt.xlabel("Step"); plt.ylabel("Gate Value")
    plt.show()

# ------------------- RUN EVERYTHING -------------------
if __name__ == "__main__":
    print("Starting full ablation experiment...")
    results = run_ablation()
    
    print("\nGeneration with Observer:")
    generate("To be or not to be", max_new=150)
    
    print("\nGeneration WITHOUT Observer (pure attention baseline):")
    generate("To be or not to be", max_new=150, use_observer=False)
    
    print("\nObserver gate collapse plot:")
    plot_gate_collapse()
