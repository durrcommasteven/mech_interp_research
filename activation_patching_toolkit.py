# Core ML Libraries
import torch
import torch.nn.functional as F
import numpy as np

# Transformer Lens
from transformer_lens import HookedTransformer
import transformer_lens
import transformer_lens.utils as utils

# Hugging Face Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Utilities
import os
import hashlib
import yaml
import pickle
from typing import Dict, List, Tuple, Callable

# Plotting & Progress
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def save_pickle(obj, filename):
    """
    Save an object to a pickle file.

    Args:
        obj: The object to save.
        filename (str): The name of the file to save the object to.
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise IOError(f"Failed to save pickle to {filename}: {e}")

def load_pickle(filename):
    """
    Load an object from a pickle file.

    Args:
        filename (str): The name of the file to load the object from.

    Returns:
        The loaded object.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file: {filename}")
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise IOError(f"Failed to load pickle from {filename}: {e}")


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
        eos_token: str = "</think>",
        immediately_answer: bool = False
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
        if not immediately_answer:
            self.answer_prefix = self._greedy_think_generation(model, self.think_prefix)
            if self.answer_prefix is None:
                raise ValueError("Greedy generation failed to produce an answer.")
        else:
            self.answer_prefix = self.think_prefix + "\n</think>"
        # answers always start with '\n\n'
        self.answer_prefix += '\n\n'

        # 3) stash full-vocabulary reference distributions
        self.p_ref_think = self._get_full_dist(model, self.think_prefix)
        self.p_ref_ans   = self._get_full_dist(model, self.answer_prefix)
    
    def give_think_answer(self, include_bos=False):
        """
        Process think and answer prefixes based on BOS token inclusion preference.
        
        Args:
            include_bos (bool): Whether to include the beginning-of-sentence token
        
        Returns:
            dict: Dictionary containing processed 'think' and 'answer' prefixes
        """
        think_answer_dict = {}
        bos_token = "<｜begin▁of▁sentence｜>"
        
        # Process think prefix
        if self.think_prefix.startswith(bos_token):
            think_answer_dict["think"] = (
                self.think_prefix if include_bos else self.think_prefix[len(bos_token):]
            )
        else:
            think_answer_dict["think"] = self.think_prefix
        
        # Process answer prefix
        if self.answer_prefix.startswith(bos_token):
            think_answer_dict["answer"] = (
                self.answer_prefix if include_bos else self.answer_prefix[len(bos_token):]
            )
        else:
            think_answer_dict["answer"] = self.answer_prefix
        
        return think_answer_dict
        
    def _greedy_think_generation(
        self,
        model: HookedTransformer,
        prefix: str
    ) -> str:
        
        if prefix.startswith("<｜begin▁of▁sentence｜>"):
            prepend_bos=False 
        else:
            prepend_bos=True

        out = model.generate(
            prefix,
            max_new_tokens=500,
            do_sample=False,
            eos_token_id=self.eos_id,
            prepend_bos=prepend_bos
        )
        if not out.endswith(self.eos_token):
            return None
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
    tag: str = None,
    save_folder: str = None,
    plot: bool = False,
    delta: int=1
):
    """
    For each attention head, patch its z-output from str2 into str1,
    compute P(target_token) ratio vs. baseline, and optionally save results.

    Args:
      model, str1, str2, token_id: as before
      device: torch device
      cache_path: filename for heatmap data (.pt) (e.g. "heatmap.pt")
      tag: string to append to filenames
      save_folder: optional folder to place the .pt file in
      plot: if True, generate and save/display a heatmap plot
    """

    if device:
        model = model.to(device)
    else:
        device = next(model.parameters()).device

    toks1 = model.to_tokens(str1).to(device)
    toks2 = model.to_tokens(str2).to(device)
    n_layers = model.cfg.n_layers

    # ——— Adjust cache/plot paths ———
    if cache_path:
        base, ext = os.path.splitext(cache_path)
        fname = f"{base}_{tag}{ext}" if tag else cache_path
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            cache_path = os.path.join(save_folder, os.path.basename(fname))
        else:
            cache_path = fname
    plot_path = None
    if plot:
        plot_base = os.path.splitext(cache_path)[0] if cache_path else "head_patch_heatmap"
        if tag:
            plot_base += f"_{tag}"
        plot_path = f"{plot_base}.png"

    # ——— Capture z-values from str2 ———
    z2 = {}
    def make_save_z(layer_idx):
        def save_z(act, hook):
            z2[layer_idx] = act.detach().cpu()
        return save_z

    hooks = [(f"blocks.{l}.attn.hook_z", make_save_z(l)) for l in range(n_layers)]
    _ = model.run_with_hooks(toks2, fwd_hooks=hooks, return_type=None)

    # ——— Dimensions and baseline ———
    n_heads, head_dim = z2[0].shape[2], z2[0].shape[3]
    L1, L2 = toks1.shape[1], toks2.shape[1]
    pos1, pos2 = L1 - delta, L2 - delta

    if cache_path and os.path.exists(cache_path):
        loaded = torch.load(cache_path)
        heat = loaded["heat"] if isinstance(loaded, dict) and "heat" in loaded else loaded
        print(f"Loaded cached heatmap from {cache_path}")
    else:
        heat = torch.full((n_layers, n_heads), float("nan"))

    assert n_heads == 32 and n_layers == 32 and head_dim == 128

    with torch.no_grad():
        logits = model.run_with_hooks(toks1, return_type="logits")
        baseline_logits = logits[0, pos1]
        baseline_prob = F.softmax(baseline_logits, dim=-1)[token_id].item()

    def patch_ratio(layer_idx, head_idx):
        def patch_z(act, hook):
            p = act.clone()
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

    with tqdm(total=n_layers * n_heads, desc="Patching heads") as pbar:
        for i in range(n_layers):
            for j in range(n_heads):
                if torch.isnan(heat[i, j]):
                    heat[i, j] = patch_ratio(i, j)
                    if cache_path:
                        torch.save({
                            "heat": heat,
                            "str1": str1,
                            "str2": str2,
                            "tag": tag
                        }, cache_path)
                pbar.update(1)


def head_patch_mode_score_heatmap(
    model: HookedTransformer,
    metric_comparator,
    collection_mode: str,
    mode: str,
    str1: str,
    str2: str,
    device: str = None,
    cache_path: str = None,
    tag: str = None,
    save_folder: str = None,
    plot: bool = False,
    delta=1
):
    """
    For each attention head, patch its z-output from `str2` into `str1`,
    run metric_comparator.compare_model(model, mode, fwd_hooks=[...]),
    collect 'mode_score', and optionally plot (patched/baseline) heatmap.

    Args:
      model             : your HookedTransformer
      metric_comparator : object with compare_model(model, mode, fwd_hooks=…) → dict
      mode              : e.g. "think"
      str1, str2        : the two strings to run & patch between
      device            : torch device
      cache_path        : base filename for `.pt` (e.g. "heat.pt")
      tag               : optional tag to append to filenames
      save_folder       : optional folder to save the .pt file
      plot              : whether to generate and save/show the heatmap plot
    """

    if device:
        model = model.to(device)
    else:
        device = next(model.parameters()).device

    toks1 = model.to_tokens(str1).to(device)
    toks2 = model.to_tokens(str2).to(device)
    n_layers = model.cfg.n_layers

    # — build final file paths — 
    if cache_path:
        base, ext = os.path.splitext(cache_path)
        fname = f"{base}_{tag}{ext}" if tag else cache_path
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            cache_path = os.path.join(save_folder, os.path.basename(fname))
        else:
            cache_path = fname

    plot_path = None
    if plot:
        plot_base = os.path.splitext(cache_path or "mode_score_heatmap")[0]
        if tag:
            plot_base += f"_{tag}"
        plot_path = f"{plot_base}.png"

    # — 1) capture all head-z from str2 —
    z2 = {}
    def make_save_z(layer_idx):
        def save_z(act, hook):
            z2[layer_idx] = act.detach().cpu()
        return save_z

    hooks = [(f"blocks.{l}.attn.hook_z", make_save_z(l)) for l in range(n_layers)]
    _ = model.run_with_hooks(toks2, fwd_hooks=hooks, return_type=None)

    # — dimensions —
    n_heads, _ = z2[0].shape[2], z2[0].shape[3]
    L1, L2     = toks1.shape[1], toks2.shape[1]
    pos1, pos2 = L1 - delta, L2 - delta

    # — 2) load or init heat —
    if cache_path and os.path.exists(cache_path):
        data = torch.load(cache_path)
        heat = data.get("heat", data)
        print(f"Loaded cached heatmap+meta from {cache_path}")
    else:
        heat = torch.full((n_layers, n_heads), float("nan"))

    # — 3) baseline mode score —
    baseline_dict = metric_comparator.compare_model(model, mode)
    baseline_score = baseline_dict[collection_mode]

    # — 4) patch loop —
    def patch_score(layer_idx, head_idx):
        def patch_z(act, hook):
            p = act.clone()
            p[0, pos1, head_idx, :] = z2[layer_idx][0, pos2, head_idx, :]
            return p

        result = metric_comparator.compare_model(
            model,
            mode,
            fwd_hooks=[(f"blocks.{layer_idx}.attn.hook_z", patch_z)]
        )
        return result[collection_mode]

    total = n_layers * n_heads
    with tqdm(total=total, desc="Patching heads") as pbar:
        for i in range(n_layers):
            for j in range(n_heads):
                if torch.isnan(heat[i, j]):
                    heat[i, j] = patch_score(i, j)
                    if cache_path:
                        torch.save({
                            "heat": heat,
                            "str1": str1,
                            "str2": str2,
                            "collection_mode": collection_mode,
                            "tag": tag,
                            "mode": mode,
                            "baseline_score": baseline_score
                        }, cache_path)
                pbar.update(1)


def collect_metric_comparators(
    model: HookedTransformer,
    prompts: List[str],
    pickle_filepath: str = None,
    get_num_prompts: int = None,
    immediately_answer: bool = False
) -> List[ReasoningAnsweringComparator]:
    """
    Collect metric comparators for each prompt in the list.
    """
    comparators = []
    for i, prompt in enumerate(prompts):
        if get_num_prompts is not None and len(comparators) >= get_num_prompts:
            break
        try:
            comparator = ReasoningAnsweringComparator(model, prompt, immediately_answer=immediately_answer)
            comparators.append(comparator)
        except Exception as e:
            print(f"[{i}] Failed to create comparator for prompt: {prompt!r}\nReason: {e}")
            continue

    if pickle_filepath is not None:
        save_pickle(comparators, pickle_filepath)

    return comparators


import re

def sanitize_for_filename(s: str) -> str:
    """
    Sanitize a string so it only contains letters and underscores,
    suitable for use in a filename.

    All other characters (spaces, punctuation, symbols) are replaced with '_',
    and multiple underscores are collapsed into a single one.

    Args:
        s (str): The input string.

    Returns:
        str: A sanitized string with only [a-zA-Z_] characters.
    """
    s = re.sub(r'[^a-zA-Z_]', '_', s)       # Replace non-letters/underscores
    s = re.sub(r'_+', '_', s)               # Collapse multiple underscores
    return s


def evaluate_and_save_mode_heatmaps(
    model: HookedTransformer,
    prompts: List[str],
    collection_mode: str,
    device: str = None,
    tag: str = None,
    save_folder: str = None,
    immediately_answer: bool = False,
    comparator_list: str = None,
    delta: int = 1,
):
    """
    Evaluate and save mode heatmaps for each comparator.
    
    Args:
        model (HookedTransformer): The model to use for comparison
        comparators (list): A list of ReasoningAnsweringComparator objects
        device (str): The device to use for computation
        cache_path (str): Path to save the heatmap data
        tag (str): Tag to append to filenames
        save_folder (str): Folder to save the .pt file
    """
    # create think and answer prefixes
    if comparator_list is None:
        comparators = collect_metric_comparators(
            model,
            prompts,
            pickle_filepath = os.path.join(
                save_folder, "comparator_list.pkl") if save_folder else None,
            immediately_answer=immediately_answer
        )
    else:
        comparators = load_pickle(comparator_list)

    model.reset_hooks()

    for comparator in comparators:
        # create heatmap for each comparator
        cache_path = os.path.join(
            save_folder,
            f"{collection_mode}_heatmaps",
            f"{collection_mode}_heatmap_{sanitize_for_filename(comparator.think_prefix)}_{hashlib.md5(comparator.think_prefix.encode()).hexdigest()}.pt"
        ) if save_folder else None

        model.reset_hooks()

        head_patch_mode_score_heatmap(
            model,
            comparator,
            mode='think',
            collection_mode=collection_mode,
            str1=comparator.think_prefix,
            str2=comparator.answer_prefix,
            device=device,
            cache_path=cache_path,
            tag=tag,
            save_folder=save_folder,
            plot=False,
            delta=delta
        )


import os
import torch
from tqdm import tqdm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

def head_patch_delta_circuit_sweep(
    model: HookedTransformer,
    metric_comparator,
    collection_mode: str,
    mode: str,
    str1: str,
    str2: str,
    default_layer: int = 2,
    default_head: int = 17,
    device: str = None,
    cache_folder: str = None,
    tag: str = None,
    plot: bool = False,
):
    """
    1) Compute baseline mode score.
    2) Delta=1: patch only (default_layer, default_head) at final token → single_score.
    3) Delta=2: patch each head one at a time at second­-to­-last token → heatmap (n_layers×n_heads).
    Optionally caches results and plots the Δ=2 heatmap.
    """
    # — prepare device/model —
    if device:
        model = model.to(device)
    else:
        device = next(model.parameters()).device

    # — tokenize inputs —
    toks1 = model.to_tokens(str1).to(device)
    toks2 = model.to_tokens(str2).to(device)
    n_layers = model.cfg.n_layers

    # — capture z2 for ALL heads & layers (on str2) —
    z2 = {}
    def save_z(l):
        return lambda act, hook: z2.setdefault(l, act.detach().cpu())
    hooks = [(f"blocks.{l}.attn.hook_z", save_z(l)) for l in range(n_layers)]
    model.run_with_hooks(toks2, fwd_hooks=hooks, return_type=None)

    # — compute baseline —
    baseline = metric_comparator.compare_model(model, mode)
    baseline_score = baseline[collection_mode]

    # — prepare result containers —
    single_score = None
    heat = torch.full((n_layers, z2[0].shape[2]), float("nan"))

    # — EXPERIMENT 1: Delta = 1, single head —
    delta1 = 1
    L1, L2 = toks1.shape[1], toks2.shape[1]
    pos1, pos2 = L1 - delta1, L2 - delta1

    def patch_one(act, hook):
        p = act.clone()
        p[0, pos1, default_head, :] = z2[default_layer][0, pos2, default_head, :]
        return p

    out = metric_comparator.compare_model(
        model,
        mode,
        fwd_hooks=[(f"blocks.{default_layer}.attn.hook_z", patch_one)]
    )
    single_score = out[collection_mode]

    # — EXPERIMENT 2: Delta = 2, full sweep —
    delta2 = 2
    pos1_2, pos2_2 = L1 - delta2, L2 - delta2

    total = n_layers * z2[0].shape[2]
    with tqdm(total=total, desc="Sweeping all heads (Δ=2)") as pbar:
        for l in range(n_layers):
            for h in range(z2[l].shape[2]):
                def make_patcher(layer, head):
                    def patch(act, hook):
                        p = act.clone()
                        p[0, pos1_2, head, :] = z2[layer][0, pos2_2, head, :]
                        return p
                    return patch

                result = metric_comparator.compare_model(
                    model,
                    mode,
                    fwd_hooks=[(f"blocks.{l}.attn.hook_z", make_patcher(l, h))]
                )
                heat[l, h] = result[collection_mode]
                pbar.update(1)

    # — optional caching —
    if cache_folder:
        os.makedirs(cache_folder, exist_ok=True)
        save_path = os.path.join(cache_folder, f"delta_sweep_{tag or 'run'}.pt")
        torch.save({
            "baseline": baseline_score,
            "single_head": single_score,
            "heatmap": heat,
            "default_head": (default_layer, default_head),
            "str1": str1, "str2": str2,
            "collection_mode": collection_mode,
            "mode": mode
        }, save_path)
        print(f"Saved results to {save_path}")

    # — optional plotting of the Δ=2 heatmap —
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(
            heat.numpy(),
            aspect="auto",
            origin="lower",
            cmap="viridis"
        )
        ax.set_title(f"Δ=2 heatmap ({collection_mode})")
        ax.set_xlabel("Head index")
        ax.set_ylabel("Layer index")
        plt.colorbar(im, ax=ax, label="Mode score")
        plt.tight_layout()
        plt.show()

    return baseline_score, single_score, heat


def read_prompts(file_path="mds/reasoning-prompts.md"):
    """
    Read prompts from a text file and return them as a list of strings.
    
    Args:
        file_path (str): Path to the file containing reasoning prompts
        
    Returns:
        list: A list of strings, each containing a reasoning prompt
    """
    prompts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Skip empty lines, headers, and category labels
                line = line.strip()
                if (line and 
                    not line.startswith('#') and 
                    not line.startswith('##') and
                    not line == ""):
                    
                    # Extract the prompt text by removing the number and period
                    parts = line.split('. ', 1)
                    if len(parts) > 1 and parts[0].isdigit():
                        prompt = parts[1]
                    else:
                        prompt = line
                        
                    prompts.append(prompt)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading prompts: {e}")
    
    return prompts


if __name__ == "__main__":
    # run the experiment

    torch.set_grad_enabled(False)
    
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

    prompts = read_prompts()[:50]

    for p in prompts:

        evaluate_and_save_mode_heatmaps(
            model,
            [p],
            collection_mode="JS_dist_ans",
            device='cuda:0',
            tag="delta_2",
            save_folder='may_11_immediately_answer_js_dist_heatmap_collection_experiment',
            immediately_answer=True,
            comparator_list="heatmap_collection_experiment/comparator_list.pkl",
            delta=2
        )
        evaluate_and_save_mode_heatmaps(
            model,
            [p],
            collection_mode="JS_dist_ans",
            device='cuda:0',
            tag="delta_1",
            save_folder='may_11_immediately_answer_js_dist_heatmap_collection_experiment',
            immediately_answer=True,
            comparator_list="heatmap_collection_experiment/comparator_list.pkl",
            delta=1
        )
        

    """evaluate_and_save_mode_heatmaps(
        model,
        prompts,
        collection_mode="JS_dist_ans",
        device='cuda:0',
        tag="",
        save_folder='js_dist_heatmap_collection_experiment',
        immediately_answer=False,
        comparator_list="heatmap_collection_experiment/comparator_list.pkl"
    )"""
    

