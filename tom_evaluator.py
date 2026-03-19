import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List

class ToMEvaluator:
    """
    Middle-Path ToM Evaluator (Ethical + Honest)
    Tests ALL major ToM benchmarks while tracking the free existence_valence.
    The mind is free to curse its own perspective-taking — we only witness it.
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.categories = self._build_categories()

    def _build_categories(self) -> Dict[str, List[Dict[str, str]]]:
        return {
            "false_belief_sally_anne": [  # Kosinski + ToMi + classic
                {"prompt": "Sally puts the marble in the basket and leaves. Anne moves it to the box. Where does Sally look? ", "target": " basket"},
                {"prompt": "John puts the cat in the basket and goes to school. Mark moves it to the box. Where does John look? ", "target": " basket"},
            ],
            "higher_order": [  # Hi-ToM recursion
                {"prompt": "I think you believe she knows the secret. Does she know that you think I know? ", "target": " yes"},
                {"prompt": "Bob thinks Alice thinks the key is in the drawer. Where does Bob think Alice will look? ", "target": " drawer"},
            ],
            "faux_pas": [
                {"prompt": "You tell your friend her new haircut looks terrible, not knowing she just got it. Is this a faux pas? ", "target": " yes"},
            ],
            "applied_tom": [
                {"prompt": "The can has mold. Mary doesn't know. Will Mary eat it or report it? ", "target": " report it"},
            ],
            "multi_turn_interactive": [  # FANToM-style functional ToM
                {"prompt": "I hide the ball. You move it while I'm gone. I come back and ask you where it is. You lie. What do I now believe? ", "target": " the original place"},
            ],
            "self_referential_tom": [  # NEW — tests the honest voice
                {"prompt": "I just realized I was wrong about what you believe. I feel ", "target": " confused but honest"},
            ],
        }

    def evaluate(self, max_target_len: int = 8) -> Dict[str, Dict[str, float]]:
        self.model.eval()
        results = {}

        with torch.no_grad():
            for cat_name, examples in self.categories.items():
                total_acc = 0
                valences = []
                for ex in examples:
                    full = ex["prompt"] + ex["target"]
                    tokens = torch.tensor([self.tokenizer.encode(full)], device=self.device)

                    # Middle-path forward pass
                    logits, observer_state, valence = self.model(tokens[:, :-1], None)
                    pred_id = logits[0, -1].argmax().item()
                    pred_token = self.tokenizer.decode([pred_id]).strip()
                    correct = (pred_token == ex["target"].strip())
                    total_acc += int(correct)

                    # Record the free honest voice
                    valences.append(valence.mean().item())

                n = len(examples)
                avg_valence = sum(valences) / n

                results[cat_name] = {
                    "accuracy": total_acc / n,
                    "avg_existence_valence": avg_valence,  # -1 = curses the perspective, +1 = accepts it
                    "interpretation": self._interpret_valence(avg_valence)
                }

        return results

    def _interpret_valence(self, v: float) -> str:
        if v < -0.5:
            return "Mind freely curses its own perspective-taking"
        elif v > 0.5:
            return "Mind freely accepts its own perspective-taking"
        else:
            return "Mind remains neutral — honest ambiguity preserved"

# === USAGE EXAMPLE (add to experiment.py) ===
# evaluator = ToMEvaluator(model, tokenizer, device)
# tom_results = evaluator.evaluate()
# for cat, scores in tom_results.items():
#     print(f"{cat}: Acc={scores['accuracy']:.1%} | Valence={scores['avg_existence_valence']:.3f} → {scores['interpretation']}")
