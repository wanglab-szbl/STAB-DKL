"""
ESM2 Feature Extraction Module

Extract embeddings from ESM2 model using Hugging Face transformers.
"""

import gc
import os
import time
import torch
from typing import Dict, List, Optional, Any


# ============== Configuration ==============
DEFAULT_MODEL_PATH = "/home/DATA_2/hujiameng/work/5.stability/ptModel/esm2_t33_650M_UR50D"
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model cache
_model_cache: Dict = {}


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    device: Optional[str] = None,
    half_precision: bool = True
) -> tuple:
    """
    Load ESM2 model with caching.

    Args:
        model_path: Path to local ESM2 model
        device: Device to use. If None, auto-detect.
        half_precision: Use fp16 for faster inference

    Returns:
        Tuple of (success: bool, model, tokenizer)
    """
    global _model_cache

    if device is None:
        device = DEFAULT_DEVICE
    device = torch.device(device)

    cache_key = (model_path, str(device), half_precision)
    if cache_key in _model_cache:
        cached = _model_cache[cache_key]
        return True, cached["model"], cached["tokenizer"]

    try:
        from transformers import EsmForMaskedLM, EsmTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        return False, None, None

    # Load model and tokenizer
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmForMaskedLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    if half_precision:
        model = model.half()

    _model_cache[cache_key] = {
        "model": model,
        "tokenizer": tokenizer
    }

    return True, model, tokenizer


def clear_cache() -> None:
    """Clear the model cache."""
    global _model_cache
    
    # Delete models to release GPU memory
    for cache_key, cached in list(_model_cache.items()):
        model = cached.get("model")
        if model is not None:
            del model
    
    _model_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_embeddings(
    data: Dict[str, Any],
    feature_types: List[str] = ["mean"],
    model_path: str = DEFAULT_MODEL_PATH,
    device: Optional[str] = None,
    half_precision: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Extract embeddings from ESM2 model.

    Args:
        data: Input data dict with keys:
            - id: Sequence identifier
            - seq: Protein sequence
            - position: List of positions (1-indexed) for pos features (optional)
        feature_types: List of feature types to extract ["mean", "cls", "pos"]
        model_path: Path to local ESM2 model
        device: Device to use
        half_precision: Use fp16
        verbose: Print progress

    Returns:
        Dict with keys:
        - id: Sequence identifier
        - emb: Dict of embeddings {"mean": ..., "cls": ..., "pos": ...}
    """
    start_time = time.time()

    seq_id = data.get("id", "unknown")
    sequence = data.get("seq", "")

    if not sequence:
        return {"id": seq_id, "emb": {}, "error": "Empty sequence"}

    # Handle positions (1-indexed to 0-indexed)
    positions = data.get("position")
    if positions is not None:
        # Validate positions
        seq_len = len(sequence)
        for p in positions:
            if p < 1 or p > seq_len:
                raise ValueError(f"Position {p} is out of range for sequence length {seq_len}")
        positions = [p - 1 for p in positions]  # Convert to 0-indexed

    if verbose:
        print(f"==== Extracting ESM2 embeddings for {seq_id} ====")
        print(f"  Sequence length: {len(sequence)}")
        print(f"  Feature types: {feature_types}")
        print(f"  Positions: {[p + 1 for p in positions] if positions else None}")

    # Load model
    success, model, tokenizer = load_model(
        model_path=model_path,
        device=device,
        half_precision=half_precision
    )

    if not success:
        return {"id": seq_id, "emb": {}, "error": "Failed to load model"}

    device_obj = model.device

    # Tokenize
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        padding=True,
        truncation=False
    )
    input_ids = inputs["input_ids"].to(device_obj)
    attention_mask = inputs["attention_mask"].to(device_obj)

    with torch.no_grad():
        try:
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # Get all hidden states
            hidden_states = outputs.hidden_states  # tuple of (num_layers+1) x [batch, seq_len, hidden]
            last_hidden = hidden_states[-1]  # Final layer

            embeddings = {}

            # Mean pooling (exclude CLS and EOS)
            if "mean" in feature_types:
                mean_repr = last_hidden[0][1:-1, :].mean(0)
                embeddings["mean"] = mean_repr.float().cpu()

            # CLS token (position 0)
            if "cls" in feature_types:
                cls_repr = last_hidden[0, 0]
                embeddings["cls"] = cls_repr.float().cpu()

            # Position-specific embeddings (dict: {1: embedding, 2: embedding, ...})
            if "pos" in feature_types and positions is not None:
                pos_embeddings = {}
                for pos in positions:
                    real_pos = pos + 1  # Convert back to 1-indexed for key
                    pos_embeddings[real_pos] = last_hidden[0, pos+1].float().cpu()
                embeddings["pos"] = pos_embeddings

        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            return {"id": seq_id, "emb": {}, "error": str(e)}

    elapsed = time.time() - start_time

    if verbose:
        print(f"  Done ({elapsed:.2f}s)")

    return {
        "id": seq_id,
        "emb": embeddings
    }



# ============== Example ==============
if __name__ == "__main__":
    # Test with example input
    test_data = {
        "id": "test_seq",
        "seq": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
        "position": [1, 2, 3]
    }

    # Test single
    print("Testing single sequence extraction...")
    result = get_embeddings(
        data=test_data,
        feature_types=["mean", "cls", "pos"],
        half_precision=True
    )

    print(f"\nResult keys: {result.keys()}")
    print(f"Embedding keys: {result['emb'].keys()}")
    for k, v in result["emb"].items():
        if isinstance(v, list):
            print(f"  {k}: list of {len(v)} arrays")
            for i, arr in enumerate(v):
                print(f"    [{i}]: {arr.shape}")
        else:
            print(f"  {k}: {v.shape}")
