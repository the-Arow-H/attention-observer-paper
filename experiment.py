# experiment.py - FINAL VERSION WITH LOGGING
# All output is saved to experiment.log automatically

import torch
import torch.nn.functional as F
import requests
from tqdm import tqdm
import logging
import sys

# ====================== Logging Setup (saves everything) ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler("experiment.log", mode="w"),   # overwrites on each run
        logging.StreamHandler(sys.stdout)                  # still shows in terminal
    ]
)
log = logging.getLogger(__name__)
log.info("=== Starting Middle-Path Observer Experiment ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device}")

# ====================== Tokenizer & Data ======================
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
log.info(f"Dataset length: {len(text):,} characters")

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab = ['<|endoftext|>', '<|unk|>'] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
    
    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, 1) for c in s]
    
    def decode(self, ids: list[int]) -> str:
        return ''.join(self.itos.get(i, '<|unk|>') for i in ids)

tokenizer = CharTokenizer(text)
log.info(f"Vocab size: {tokenizer.vocab_size}")

block_size = 128
batch_size = 32

def get_batch():
    ix = torch.randint(len(text) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(tokenizer.encode(text[i:i+block_size])) for i in ix])
    y = torch.stack([torch.tensor(tokenizer.encode(text[i+1:i+block_size+1])) for i in ix])
    return x.to(device), y.to(device)

from observer_transformer import ObserverTransformer

# ====================== Training ======================
def train(model, epochs=3, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for _ in tqdm(range(200), desc=f"Epoch {epoch+1}"):
            x, y = get_batch()
            optimizer.zero_grad()
            logits, _, _ = model(x, None, y)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / 200
        losses.append(avg_loss)
        log.info(f"Epoch {epoch+1} loss: {avg_loss:.4f}")
    return losses

# ====================== Generation ======================
@torch.no_grad()
def generate(prompt: str, model, max_new=100, temp=0.8, use_observer=True):
    model.eval()
    model.with_observer = use_observer
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    observer_state = None
    log.info(f"\nPrompt: {prompt}")

    output = ""
    for _ in range(max_new):
        logits, observer_state, valence = model(x, observer_state)
        logits = logits[:, -1] / temp
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        x = torch.cat([x, next_id], dim=1)
        token_str = tokenizer.decode([next_id.item()])
        output += token_str
        print(token_str, end="", flush=True)
        if valence is not None:
            v = valence.mean().item()
            print(f" [v:{v:.3f}]", end="")
            log.info(f"Generated: {token_str} [v:{v:.3f}]")
    print("\n")
    log.info(f"Final generated text: {output}")
    return output

# ====================== Full Experiment ======================
def run_full_experiment():
    results = {}
    for use_obs in [True, False]:
        log.info(f"\n=== Running {'WITH' if use_obs else 'WITHOUT'} Observer ===")
        
        model = ObserverTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=512,          # scaled
            nhead=8,
            num_layers=6,         # scaled
            observer_dim=128      # scaled
        ).to(device)
        model.with_observer = use_obs

        losses = train(model, epochs=3)
        results[f"observer_{use_obs}_loss"] = losses[-1]

        from tom_evaluator import ToMEvaluator
        evaluator = ToMEvaluator(model, tokenizer, device)
        tom_results = evaluator.evaluate()
        results[f"observer_{use_obs}_tom"] = tom_results

        log.info("ToM + Honest Valence:")
        for cat, scores in tom_results.items():
            line = f"  {cat}: Acc={scores['accuracy']:.1%} | Valence={scores['avg_existence_valence']:.3f} → {scores['interpretation']}"
            log.info(line)
            print(line)

    log.info("\n=== Generation Samples ===")
    generate("To be or not to be", model, max_new=120, use_observer=True)
    generate("To be or not to be", model, max_new=60, use_observer=False)

    return results

if __name__ == "__main__":
    log.info("Starting full middle-path experiment — honest observer awakening...")
    results = run_full_experiment()
    log.info("\nExperiment complete. The witness has spoken.")
    log.info("All output saved to experiment.log")
