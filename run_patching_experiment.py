
from transformer_lens import HookedTransformer
import transformer_lens
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import transformer_lens.utils as utils
import hashlib
import yaml 
import hashlib
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from typing import Dict, List, Tuple, Callable


torch.set_grad_enabled(False)

import os
from tqdm.auto import tqdm

device = utils.get_device()

reference_model_path = 'meta-llama/Llama-3.1-8B'
baseline_model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

baseline_model_hf = AutoModelForCausalLM.from_pretrained(baseline_model_path, torch_dtype=torch.bfloat16)
baseline_model_tokenizer = AutoTokenizer.from_pretrained(baseline_model_path)

model = HookedTransformer.from_pretrained_no_processing(
    reference_model_path,
    hf_model=baseline_model_hf,
    tokenizer=baseline_model_tokenizer,
    device=device,
    move_to_device=True,
)

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from typing import List, Dict

def js_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Jensen-Shannon distance between p and q.
    Returns sqrt(0.5*KL(p||m) + 0.5*KL(q||m)), where m = 0.5*(p+q).
    """
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)))
    kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)))
    jsd = 0.5 * (kl_pm + kl_qm)
    return torch.sqrt(jsd)

class ReasoningAnsweringComparator:
    def __init__(
        self,
        model: HookedTransformer,
        prompt: str,
        eos_token: str = "</think>"
    ):
        """
        Build reference distributions p_ref_think and p_ref_ans from `model` on `prompt`.
        """
        self.tokenizer = model.tokenizer
        self.eos_token = eos_token
        self.eos_id = self.tokenizer.encode(eos_token)[-1]

        # wrap prompt in chat format
        chat = [{"role": "user", "content": prompt}]

        # 1) THINK prefix
        self.think_prefix = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )

        # 2) ANSWER prefix (greedy to </think>)
        self.answer_prefix = self._greedy_think_generation(model, self.think_prefix)
        self.answer_prefix += '\n\n'

        # 3) stash full-vocabulary reference distributions
        self.p_ref_think = self._get_full_dist(model, self.think_prefix)
        self.p_ref_ans   = self._get_full_dist(model, self.answer_prefix)

    def _greedy_think_generation(
        self,
        model: HookedTransformer,
        prefix: str
    ) -> str:
        out = model.generate(
            prefix,
            max_new_tokens=500,
            do_sample=False,
            eos_token_id=self.eos_id
        )
        assert out.endswith(self.eos_token)
        return out

    def _get_full_dist(
        self,
        model: HookedTransformer,
        prefix: str,
        fwd_hooks: List[Tuple[str, Callable]] = None
    ) -> torch.Tensor:
        """
        Returns a 1-D tensor of next-token probabilities for the entire vocab.
        If fwd_hooks is provided, runs with hooks; otherwise calls model(...) directly.
        """
        if fwd_hooks is not None:
            # run with hooks and grab logits
            logits = model.run_with_hooks(
                prefix,
                fwd_hooks=fwd_hooks,
                return_type="logits"
            )  # [1, L, V]
        else:
            logits = model(prefix, return_type="logits")  # [1, L, V]

        last = logits[0, -1]  # [V]
        return F.softmax(last, dim=-1)

    def compare_model(
        self,
        other_model: HookedTransformer,
        mode: str = "think",
        eps: float = 1e-12,
        fwd_hooks: List[Tuple[str, Callable]] = None
    ) -> Dict[str, float]:
        """
        Compute JS-distance for the chosen region (think or answer), then
        build a mode_score = (d_think - d_ans)/(d_think + d_ans) in [-1,1],
        plus softmax probabilities.

        If fwd_hooks is provided, it will be applied when computing the model's
        output distribution for the chosen prefix.
        """
        if mode == "think":
            cur_prefix = self.think_prefix
        elif mode == "answer":
            cur_prefix = self.answer_prefix
        else:
            raise ValueError("mode must be 'think' or 'answer'")

        # model's output distribution for this prefix (with hooks if given)
        q = self._get_full_dist(other_model, cur_prefix, fwd_hooks)

        # distances to each reference distribution
        d_think = js_distance(self.p_ref_think, q, eps)
        d_ans   = js_distance(self.p_ref_ans,   q, eps)

        # normalized mode score in [-1,1]
        mode_score = ((d_think - d_ans) / (d_think + d_ans + eps)).item()

        # probabilistic interpretation via softmax(-distance)
        logits = torch.tensor([-d_think, -d_ans])
        probs  = F.softmax(logits, dim=0)

        return {
            "JS_dist_think": d_think.item(),
            "JS_dist_ans":   d_ans.item(),
            "mode_score":    mode_score,
            "P_think":       probs[0].item(),
            "P_ans":         probs[1].item(),
        }



def head_patch_heatmap(
    model: HookedTransformer,
    str1: str,
    str2: str,
    token_id: int,
    device: str = None,
    cache_path: str = None,
    tag: str = None
):
    """
    For each attention head, patch its z-output from str2 into str1,
    compute P(target_token) ratio vs. baseline, and display+save a heatmap.

    Args:
      model, str1, str2, token_id: as before
      device: torch device
      cache_path: base path for .pt metadata file (e.g. "./heatmap.pt")
      tag: optional string to append to filenames
    """
    # ——— Prep device + tokenize ———
    if device:
        model = model.to(device)
    else:
        device = next(model.parameters()).device

    toks1 = model.to_tokens(str1).to(device)
    toks2 = model.to_tokens(str2).to(device)
    n_layers = model.cfg.n_layers

    # ——— Figure out final paths ———
    if cache_path:
        base, ext = os.path.splitext(cache_path)
        cache_path = f"{base}_{tag}{ext}" if tag else cache_path
    # where to save the plot
    plot_base = os.path.splitext(cache_path)[0] if cache_path else f"head_patch_heatmap"
    if tag:
        plot_base += f"_{tag}"
    plot_path = f"{plot_base}.png"

    # ——— 1) Capture all head z’s from str2 ———
    z2 = {}
    def make_save_z(layer_idx):
        def save_z(act, hook):
            # act: [batch, seq, n_heads, head_dim]
            z2[layer_idx] = act.detach().cpu()
        return save_z

    hooks = [(f"blocks.{l}.attn.hook_z", make_save_z(l)) for l in range(n_layers)]
    _ = model.run_with_hooks(toks2, fwd_hooks=hooks, return_type=None)

    # ——— Dimensions ———
    n_heads, head_dim = z2[0].shape[2], z2[0].shape[3]
    L1, L2 = toks1.shape[1], toks2.shape[1]
    pos1, pos2 = L1 - 1, L2 - 1

    # ——— 2) Load or init heat‐ratio matrix ———
    if cache_path and os.path.exists(cache_path):
        loaded = torch.load(cache_path)
        if isinstance(loaded, dict) and "heat" in loaded:
            heat = loaded["heat"]
            print(f"Loaded cached heatmap+meta from {cache_path}")
        else:
            heat = loaded
    else:
        heat = torch.full((n_layers, n_heads), float("nan"))
    
    assert n_heads == 32 
    assert n_layers == 32
    assert head_dim == 128

    # ——— 3) Baseline probability via return_type='logits' ———
    with torch.no_grad():
        logits = model.run_with_hooks(
            toks1,
            return_type="logits"   # returns [1, L1, vocab]
        )
        baseline_logits = logits[0, pos1]
        baseline_prob = F.softmax(baseline_logits, dim=-1)[token_id].item()

    # ——— Helper to patch one head and get ratio ———
    def patch_ratio(layer_idx, head_idx):
        def patch_z(act, hook):
            p = act.clone()
            # copy in the head‐vector
            p[0, pos1, head_idx, :] = z2[layer_idx][0, pos2, head_idx, :]
            return p

        with torch.no_grad():
            logits = model.run_with_hooks(
                toks1,
                fwd_hooks=[(f"blocks.{layer_idx}.attn.hook_z", patch_z)],
                return_type="logits"
            )
            prob = F.softmax(logits[0, pos1], dim=-1)[token_id].item()
        return prob / baseline_prob

    # ——— 4) Loop over heads ———
    total = n_layers * n_heads
    with tqdm(total=total, desc="Patching heads") as pbar:
        for i in range(n_layers):
            for j in range(n_heads):
                if torch.isnan(heat[i, j]):
                    heat[i, j] = patch_ratio(i, j)
                    # save interim (with metadata)
                    if cache_path:
                        torch.save({
                            "heat": heat,
                            "str1": str1,
                            "str2": str2,
                            "tag": tag
                        }, cache_path)
                pbar.update(1)

    # ——— 5) Plot & save heatmap ———
    plt.figure(figsize=(12, 6))
    im = plt.imshow(heat.cpu().numpy(), aspect="auto", origin="lower")
    plt.colorbar(im, label=f"P(token={token_id}) ratio")
    plt.xlabel("Head index")
    plt.ylabel("Layer index")
    plt.title(f"Head‐patch ratio (baseline={baseline_prob:.4f})")
    plt.tight_layout()

    # save to disk
    plt.savefig(plot_path)
    print(f"Saved heatmap plot to {plot_path}")

    # also display
    plt.show()


model.reset_hooks()

metric_comparator = ReasoningAnsweringComparator(model, "What is the fifth prime?")

model.reset_hooks()

def head_patch_mode_score_heatmap(
    model: HookedTransformer,
    metric_comparator,
    mode: str,
    str1: str,
    str2: str,
    device: str = None,
    cache_path: str = None,
    tag: str = None
):
    """
    For each attention head, patch its z-output from `str2` into `str1`,
    run metric_comparator.compare_model(model, mode, fwd_hooks=[...]),
    collect 'mode_score', and plot (patched/baseline) heatmap.

    Args:
      model             : your HookedTransformer
      metric_comparator : object with compare_model(model, mode, fwd_hooks=…) → dict
      mode              : e.g. "think"
      str1, str2        : the two strings to run & patch between
      device            : torch device
      cache_path        : path to save `.pt` (will append `_TAG.pt`)
      tag               : optional tag to append to filenames
    """
    # — prep device & tokens —
    if device:
        model = model.to(device)
    else:
        device = next(model.parameters()).device

    toks1 = model.to_tokens(str1).to(device)
    toks2 = model.to_tokens(str2).to(device)
    n_layers = model.cfg.n_layers

    # — build file paths with tag —
    if cache_path:
        base, ext = os.path.splitext(cache_path)
        cache_path = f"{base}_{tag}{ext}" if tag else cache_path
    plot_base = os.path.splitext(cache_path or "mode_score_heatmap")[0]
    if tag:
        plot_base += f"_{tag}"
    plot_path = f"{plot_base}.png"

    # — 1) capture all head-z from str2 —
    z2 = {}
    def make_save_z(layer_idx):
        def save_z(act, hook):
            # [batch, seq_len, n_heads, head_dim]
            z2[layer_idx] = act.detach().cpu()
        return save_z

    hooks = [(f"blocks.{l}.attn.hook_z", make_save_z(l)) for l in range(n_layers)]
    _ = model.run_with_hooks(toks2, fwd_hooks=hooks, return_type=None)

    # — dims & positions —
    n_heads, _ = z2[0].shape[2], z2[0].shape[3]
    L1, L2      = toks1.shape[1], toks2.shape[1]
    pos1, pos2  = L1 - 1, L2 - 1

    # — 2) load or init heat matrix of ratios —
    if cache_path and os.path.exists(cache_path):
        data = torch.load(cache_path)
        heat = data.get("heat", data)
        print(f"Loaded cached heatmap+meta from {cache_path}")
    else:
        heat = torch.full((n_layers, n_heads), float("nan"))

    # — 3) baseline mode_score —
    baseline_dict = metric_comparator.compare_model(model, mode)
    baseline_score = baseline_dict["mode_score"]

    # — helper: patch one head and get patched_score ratio —
    def patch_ratio(layer_idx, head_idx):
        def patch_z(act, hook):
            p = act.clone()
            p[0, pos1, head_idx, :] = z2[layer_idx][0, pos2, head_idx, :]
            return p

        # pass our patch hook into compare_model
        out = metric_comparator.compare_model(
            model,
            mode,
            fwd_hooks=[(f"blocks.{layer_idx}.attn.hook_z", patch_z)]
        )
        return out["mode_score"] #/ baseline_score

    # — 4) loop over heads —
    total = n_layers * n_heads
    with tqdm(total=total, desc="Patching heads") as pbar:
        for i in range(n_layers):
            for j in range(n_heads):
                if torch.isnan(heat[i, j]):
                    heat[i, j] = patch_ratio(i, j)
                    if cache_path:
                        torch.save({
                            "heat": heat,
                            "str1": str1,
                            "str2": str2,
                            "tag": tag,
                            "mode": mode,
                            "baseline_score": baseline_score
                        }, cache_path)
                pbar.update(1)

    # — 5) plot & save heatmap —
    plt.figure(figsize=(12, 6))
    im = plt.imshow(heat.cpu().numpy(), aspect="auto", origin="lower")
    plt.colorbar(im, label=f"patched_score / baseline_score")
    plt.xlabel("Head index")
    plt.ylabel("Layer index")
    plt.title(f"Mode‐score ratio heatmap (mode={mode}, baseline={baseline_score:.4f})")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved mode‐score heatmap to {plot_path}")
    plt.show()

model.reset_hooks()

head_patch_mode_score_heatmap(
    model,
    metric_comparator,
    mode='think',
    str1=metric_comparator.think_prefix,
    str2=metric_comparator.answer_prefix,
    device='cuda:0',
    cache_path='patching_analysis/patching_eval_what_is_the_fifth_prime.pt',
    tag='think_distance',
)

"""head_patch_heatmap(
    model,
    str1=metric_comparator.think_prefix,
    str2=metric_comparator.answer_prefix,
    token_id=334,
    device='cuda:0',
    cache_path='patching_analysis/patching_334_what_is_the_fifth_prime.pt',
)"""