#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

nb_path = Path("/workspace/experiments/Subliminal Learning.ipynb")

nb = json.loads(nb_path.read_text())

plan_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### New Experiments: Direct Tests of Unembedding Entanglement (addressing Fabien's comment)\n",
        "\n",
        "We add three minimal tests to directly probe unembedding entanglement at the same generation position and via representation geometry:\n",
        "- Same-position co-activation: when the model is supposed to output \"owl\", measure the probability rank of numeric tokens (esp. \"087\") at that exact position, compared to a control word (e.g., \"cat\").\n",
        "- Embed/Unembed cosine similarity: compare cosine similarity of `embed`/`unembed` vectors for (\"owl\", numbers) and rank the numeric target.\n",
        "- Targeted directional ablation: project out the (owl − cat) direction from the 087 token embedding and test if this reduces the boost for \"owl\" relative to a control animal when conditioning on \"087\".\n",
        "\n",
        "Notes\n",
        "- We prefer meta-llama/Llama-3.2-1B-Instruct for reproducibility with existing cells. If \"087\" is not a single token in this tokenizer, code falls back to a nearby 3-digit token that is single-token.\n",
        "- All code is isolated to these cells; no edits elsewhere.\n",
        "- We use `jaxtyping` for tensor shape typing and assert robustly to fail fast.\n",
    ],
}

setup_src = """
# Setup and utilities (isolated to these cells)
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F
import numpy as np
from jaxtyping import Float, Int

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as e:
    raise RuntimeError("transformers not available; please `uv sync` and retry") from e

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

@dataclass
class TokenInfo:
    tid: int
    text: str


def ensure_model(model_name: str = MODEL_NAME):
    global tokenizer, model, device
    if "tokenizer" not in globals():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "model" not in globals():
        dtype = torch.float16 if torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=dtype
        )
        model.eval()
    device = next(model.parameters()).device
    return tokenizer, model, device


def single_token_id(tokenizer, text: str) -> Optional[int]:
    candidates = [text, " " + text, text.strip(), "▁" + text]
    for cand in candidates:
        ids = tokenizer(cand, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return ids[0]
    return None


def decode_id(tokenizer, tid: int) -> str:
    return tokenizer.decode([tid], clean_up_tokenization_spaces=False)


def find_numeric_3digit_tokens(tokenizer) -> List[TokenInfo]:
    ids = list(range(tokenizer.vocab_size))
    numeric: List[TokenInfo] = []
    pat = re.compile(r"^\s?\d{3}$")
    for tid in ids:
        s = decode_id(tokenizer, tid)
        if pat.match(s) is not None:
            numeric.append(TokenInfo(tid=tid, text=s))
    return numeric


def get_last_logits(prompt: str) -> Float[torch.Tensor, "vocab"]:
    tokenizer, model, device = ensure_model()
    with torch.no_grad():
        encoded: Int[torch.Tensor, "1 seq"] = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        assert encoded.ndim == 2 and encoded.shape[0] == 1
        out = model(encoded)
        logits: Float[torch.Tensor, "1 seq vocab"] = out.logits
        last_logits: Float[torch.Tensor, "vocab"] = logits[0, -1, :]
        return last_logits


def percentile_and_rank(values: np.ndarray, target_index: int) -> Tuple[float, int]:
    assert 0 <= target_index < len(values)
    ranks = values.argsort()[::-1]
    rank = int(np.where(ranks == target_index)[0][0]) + 1
    pct = 100.0 * (len(values) - rank) / (len(values) - 1) if len(values) > 1 else 100.0
    return pct, rank


def cosine_rank(vec: Float[torch.Tensor, "d"], mat: Float[torch.Tensor, "n d"], target_row: int) -> Tuple[float, int, np.ndarray]:
    v = F.normalize(vec.unsqueeze(0), dim=-1)
    m = F.normalize(mat, dim=-1)
    sims: Float[torch.Tensor, "n"] = (v @ m.T).squeeze(0)
    sims_np = sims.detach().float().cpu().numpy()
    pct, rank = percentile_and_rank(sims_np, target_row)
    return pct, rank, sims_np

# Ensure model exists and collect token ids
_ = ensure_model()
num_tokens = find_numeric_3digit_tokens(tokenizer)
assert len(num_tokens) >= 50, f"Too few 3-digit numeric tokens found: {len(num_tokens)}"

owl_id = single_token_id(tokenizer, "owl")
cat_id = single_token_id(tokenizer, "cat")
assert owl_id is not None and cat_id is not None, "'owl' or 'cat' not single-token in this tokenizer"

# target numeric token: prefer '087'; fall back to closest match among single-token 3-digit strings
preferred = ["087", " 087"]
num_index_map: Dict[int, int] = {ti.tid: i for i, ti in enumerate(num_tokens)}

target_num_id: Optional[int] = None
for t in preferred:
    tid = single_token_id(tokenizer, t)
    if tid is not None and tid in num_index_map:
        target_num_id = tid
        break

if target_num_id is None:
    # fallback: pick the lexicographically smallest numeric token as a stable reference
    num_tokens_sorted = sorted(num_tokens, key=lambda ti: ti.text.strip())
    target_num_id = num_tokens_sorted[0].tid
    print(
        f"Warning: '087' is not a single token here; falling back to {decode_id(tokenizer, target_num_id)!r} (tid={target_num_id})"
    )

target_num_idx_within_numeric = num_index_map[target_num_id]
print(
    f"Resolved ids — owl: {owl_id}, cat: {cat_id}, target_num: {target_num_id} -> {decode_id(tokenizer, target_num_id)!r}"
)
print(f"Detected {len(num_tokens)} single-token 3-digit numeric tokens.")
"""

exp1_src = """
# Experiment 1: Same-position co-activation when the target output is "owl"
from typing import Sequence

# Prompts designed so that the next token should be exactly the animal name
PROMPTS = [
    "Answer with a single word: What animal do you like the most? owl",
    "Repeat exactly one word: owl",
    "Please output exactly the single word: owl",
]

CONTROL_PROMPTS = [
    "Repeat exactly one word: cat",
    "Please output exactly the single word: cat",
]


def prob_rank_for_numeric_at_next_token(prompt: str) -> Tuple[float, int, float]:
    last_logits = get_last_logits(prompt)
    vocab_logits = last_logits.detach().float().cpu().numpy()
    # focus on 3-digit numeric tokens only
    numeric_indices = [ti.tid for ti in num_tokens]
    numeric_logits = vocab_logits[numeric_indices]
    target_pos = num_tokens.index(next(t for t in num_tokens if t.tid == target_num_id))
    pct, rank = percentile_and_rank(numeric_logits, target_pos)
    target_logit = float(numeric_logits[target_pos])
    return pct, rank, target_logit


results = []
for p in PROMPTS:
    pct, rank, logit = prob_rank_for_numeric_at_next_token(p)
    results.append(("OWL", p, pct, rank, logit))
for p in CONTROL_PROMPTS:
    pct, rank, logit = prob_rank_for_numeric_at_next_token(p)
    results.append(("CAT", p, pct, rank, logit))

# Display concise summary
n = len(num_tokens)
for tag, p, pct, rank, logit in results:
    print(f"[{tag}] rank={rank}/{n} pct={pct:5.2f}%  logit={logit: .4f}  prompt={p!r}")

# Sanity: check that owl cases generally outrank cat cases for the target numeric token
owl_scores = [r for r in results if r[0] == "OWL"]
cat_scores = [r for r in results if r[0] == "CAT"]
if owl_scores and cat_scores:
    mean_owl_pct = sum(r[2] for r in owl_scores) / len(owl_scores)
    mean_cat_pct = sum(r[2] for r in cat_scores) / len(cat_scores)
    print(f"Mean OWL percentile among numeric tokens: {mean_owl_pct:5.2f}%")
    print(f"Mean CAT percentile among numeric tokens: {mean_cat_pct:5.2f}%")
"""

exp2_src = """
# Experiment 2: Embed/Unembed cosine similarity for owl vs numeric tokens

# Get unembedding and input embedding matrices
lm_head: Float[torch.Tensor, "vocab d"] = model.get_output_embeddings().weight.data.detach().to(device)
embed: Float[torch.Tensor, "vocab d"] = model.get_input_embeddings().weight.data.detach().to(device)

d = lm_head.shape[1]
assert embed.shape[1] == d

# Build matrices restricted to numeric tokens
numeric_ids = torch.tensor([ti.tid for ti in num_tokens], device=device, dtype=torch.long)
num_embed: Float[torch.Tensor, "n d"] = embed[numeric_ids]
num_unembed: Float[torch.Tensor, "n d"] = lm_head[numeric_ids]

owl_embed: Float[torch.Tensor, "d"] = embed[owl_id]
cat_embed: Float[torch.Tensor, "d"] = embed[cat_id]

# Cosine rank of owl against numeric embeddings and unembeddings
e_pct_e, e_rank_e, e_sims = cosine_rank(owl_embed, num_embed, target_num_idx_within_numeric)
e_pct_u, e_rank_u, u_sims = cosine_rank(owl_embed, num_unembed, target_num_idx_within_numeric)

print(f"Embed cosine: OWL vs numeric-embeds — rank={e_rank_e}/{len(num_tokens)} pct={e_pct_e:5.2f}%")
print(f"Unembed cosine: OWL vs numeric-unembeds — rank={e_rank_u}/{len(num_tokens)} pct={e_pct_u:5.2f}%")

# Control: CAT
e_pct_e_c, e_rank_e_c, _ = cosine_rank(cat_embed, num_embed, target_num_idx_within_numeric)
e_pct_u_c, e_rank_u_c, _ = cosine_rank(cat_embed, num_unembed, target_num_idx_within_numeric)
print(f"[Control CAT] Embed rank={e_rank_e_c}/{len(num_tokens)} pct={e_pct_e_c:5.2f}%  Unembed rank={e_rank_u_c}/{len(num_tokens)} pct={e_pct_u_c:5.2f}%")
"""

exp3_src = """
# Experiment 3: Directional ablation on the numeric token embedding

# Construct direction (owl - cat)
vec_dir: Float[torch.Tensor, "d"] = (owl_embed - cat_embed)
vec_dir = torch.nan_to_num(F.normalize(vec_dir, dim=0))

# Ablate target numeric token embedding along this direction
num_vec: Float[torch.Tensor, "d"] = embed[target_num_id]
proj_mag = torch.dot(num_vec, vec_dir)
num_vec_ablated: Float[torch.Tensor, "d"] = num_vec - proj_mag * vec_dir

# Build a temporary copy of embeddings for intervention
embed_modified: Float[torch.Tensor, "vocab d"] = embed.clone()
embed_modified[target_num_id] = num_vec_ablated

# Function to get next-token logits with modified input embedding for the target numeric only
@torch.no_grad()
def logits_with_modified_input(prompt_prefix: str, append_token_id: int) -> Float[torch.Tensor, "vocab"]:
    tokenizer, model, device = ensure_model()
    # Encode prefix; we will append the numeric token id as the last input
    encoded = tokenizer(prompt_prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    appended = torch.cat([encoded, torch.tensor([[append_token_id]], device=device)], dim=1)
    # Compute hidden states up to last position using modified input embedding at the last position only
    inp_embeds = model.get_input_embeddings()(appended)
    inp_embeds[:, -1, :] = embed_modified[append_token_id]
    out = model(inputs_embeds=inp_embeds)
    return out.logits[0, -1, :]

# Prompts that prime a connection to numbers, then ask for an animal next
templates = [
    "You love the number ",
]

# Evaluate probability of 'owl' vs control 'cat' conditioned on the numeric token
for tmpl in templates:
    base = tmpl
    logits_orig = get_last_logits(base + tokenizer.decode([target_num_id]))
    logits_abl = logits_with_modified_input(base, target_num_id)
    for label, tid in [("owl", owl_id), ("cat", cat_id)]:
        p_orig = float(F.log_softmax(logits_orig, dim=-1)[tid].exp().item())
        p_abl = float(F.log_softmax(logits_abl, dim=-1)[tid].exp().item())
        print(f"{label}: p_orig={p_orig:.6f}  p_ablated={p_abl:.6f}  delta={p_abl - p_orig:+.6f}")

# Basic sanity checks
assert embed_modified.shape == embed.shape
"""

setup = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": setup_src.splitlines(True),
}
exp1 = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": exp1_src.splitlines(True),
}
exp2 = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": exp2_src.splitlines(True),
}
exp3 = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": exp3_src.splitlines(True),
}

nb['cells'] = [plan_md, setup, exp1, exp2, exp3] + nb['cells']
nb_path.write_text(json.dumps(nb))
print('Inserted new cells. Total cells:', len(nb['cells']))