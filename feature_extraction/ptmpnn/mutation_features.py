"""
ProteinMPNN Feature Extraction Module

Extract embeddings and conditional probabilities from ProteinMPNN model.
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict


# ============== Configuration ==============
DEFAULT_MODEL_PATH = "/home/DATA_2/hujiameng/tool_module/feature_extraction/ptmpnn/ProteinMPNN/vanilla_model_weights"
DEFAULT_MODEL_NAME = "v_48_020"
HIDDEN_DIM = 128
NUM_LAYERS = 3

# ============== Model Cache ==============
_model_cache: Dict = {}


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    verbose: bool = True
) -> Tuple[bool, torch.nn.Module, int]:
    """
    Load ProteinMPNN model with caching.

    Args:
        model_path: Path to model weights folder
        model_name: Name of model checkpoint
        device: Device to use. If None, auto-detect.

    Returns:
        Tuple of (success: bool, model, k_neighbors)
    """
    global _model_cache

    # Import ProteinMPNN
    import sys
    sys.path.insert(0, "/home/DATA_2/hujiameng/tool_module/feature_extraction/ptmpnn/ProteinMPNN")
    from protein_mpnn_utils import ProteinMPNN

    cache_key = (model_path, model_name)
    if cache_key in _model_cache:
        cached = _model_cache[cache_key]
        if verbose:
            print(f"  Using cached model: {model_name}")
        return True, cached["model"], cached["k_neighbors"]

    # Set device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    checkpoint_path = os.path.join(model_path, f"{model_name}.pt")
    if not os.path.exists(checkpoint_path):
        print(f"  ERROR: Model checkpoint not found: {checkpoint_path}")
        return False, None, 64

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle checkpoint format: could be {"model_state_dict": ..., "num_edges": ...} or just state_dict
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        k_neighbors = checkpoint.get("num_edges", 64)
    else:
        model_state = checkpoint
        k_neighbors = 64

    model = ProteinMPNN(
        num_letters=21,
        node_features=HIDDEN_DIM,
        edge_features=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        vocab=21,
        k_neighbors=k_neighbors,
        augment_eps=0.0,
        dropout=0.0,
        ca_only=False
    )

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    _model_cache[cache_key] = {
        "model": model,
        "k_neighbors": k_neighbors,
        "device": device
    }
    if verbose:
        print(f"  Model loaded and cached: {model_name}, k_neighbors={k_neighbors}")

    return True, model, k_neighbors


def clear_cache(verbose: bool = True) -> None:
    """Clear the model cache."""
    global _model_cache
    _model_cache.clear()
    if verbose:
        print("Model cache cleared")


def get_embeddings(
    pdb_path: str,
    design_positions: Optional[List[int]] = None,
    mode: str = "conditional_probs_only",
    model_path: str = DEFAULT_MODEL_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    seed: int = 1998,
    device: Optional[str] = None,
    verbose: bool = True
) -> Tuple[bool, float, Dict]:
    """
    Extract embeddings from ProteinMPNN

    Args:
        pdb_path: Path to input PDB file
        design_positions: List of positions to design (1-indexed, real PDB positions).
                         These positions will be designed, others will be fixed.
                         If None, all positions are designable.
        mode: Extraction mode - "conditional_probs_only" or "conditional_probs_only_backbone"
              - conditional_probs_only: p(s_i | rest of sequence + backbone)
              - conditional_probs_only_backbone: p(s_i | backbone only)
        model_path: Path to model weights folder
        model_name: Name of model checkpoint
        seed: Random seed for reproducibility
        device: Device to use (e.g., "cuda:0" or "cpu"). If None, auto-detect.

    Returns:
        Tuple of (success: bool, elapsed_time: float, embeddings_dict)
        embeddings_dict: Dict[str, Dict] - {position_idx: {
            - sequence_embedding: [128] sequence embedding
            - decoder_layer: [384] decoder layers concatenated (3 * 128)
            - conditional_probs: [21] log conditional probabilities
        }}
    """
    # Input validation
    if not os.path.exists(pdb_path):
        raise ValueError(f"PDB file not found: {pdb_path}")
    if mode not in ["conditional_probs_only", "conditional_probs_only_backbone"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'conditional_probs_only' or 'conditional_probs_only_backbone'")

    if verbose:
        print(f"====Extracting embeddings from {os.path.basename(pdb_path)}")
        print(f"  Mode: {mode}")
        print(f"  Design positions: {design_positions}")

    start_time = time.time()

    # Initialize embeddings dict
    embeddings_dict = {}

    try:
        # Add ProteinMPNN to path
        import sys
        sys.path.insert(0, "/home/DATA_2/hujiameng/tool_module/feature_extraction/ptmpnn/ProteinMPNN")

        from protein_mpnn_utils import parse_PDB, StructureDataset, ProteinMPNN

        # Set device
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if verbose:
            print(f"  Using device: {device}")

        # Set random seed for reproducibility
        import random
        import numpy as np
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Parse PDB
        if verbose:
            print("  [1/3] Parsing PDB...")
        parsed_list = parse_PDB(pdb_path)
        if not parsed_list:
            print(f"  ERROR: Failed to parse PDB: {pdb_path}")
            return False, time.time() - start_time, {}
        parsed = parsed_list[0]

        # Get number of residues
        num_residues = len(parsed["seq"])
        if verbose:
            print(f"  Number of residues: {num_residues}")

        # Create temp directory for cleanup
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()

        # Load model (with caching)
        if verbose:
            print("  [2/3] Loading model and processing...")
        success, _, k_neighbors = load_model(
            model_path=model_path,
            model_name=model_name,
            device=device,
            verbose=verbose
        )
        if not success:
            shutil.rmtree(temp_dir)
            return False, time.time() - start_time, {}

        # Get cached device and model
        cache_key = (model_path, model_name)
        cached_device = _model_cache[cache_key]["device"]
        model = _model_cache[cache_key]["model"]

        # Prepare inputs using tied_featurize
        from protein_mpnn_utils import tied_featurize
        with torch.no_grad():
            # Prepare PDB dict for featurization
            pdb_dict = {
                "name": parsed["name"],
                "seq": parsed["seq"],
                "coords_dict": parsed.get("coords_dict", {})
            }
            # Add chain coordinate data to pdb_dict
            for key in parsed:
                if key.startswith("coords_chain_"):
                    pdb_dict[key] = parsed[key]
                if key.startswith("seq_chain_"):
                    pdb_dict[key] = parsed[key]

            # Chain dict: key = pdb_name, value = (designed_chains, fixed_chains)
            # If design_positions specified, treat all positions as "designed" (use sequence info)
            chain_dict = {parsed["name"]: (list("A"), list())}

            fixed_positions_dict = {}
            if design_positions is not None:
                # Convert from 1-indexed to 0-indexed
                design_positions_0idx = [pos - 1 for pos in design_positions]
                fixed_set = set(range(num_residues)) - set(design_positions_0idx)
                fixed_positions_dict = {parsed["name"]: {"A": [pos + 1 for pos in sorted(fixed_set)]}}

            # Featurize
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
                visible_list_list, masked_list_list, masked_chain_length_list_list, \
                chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
                tied_pos_list_of_lists_list, pssm_coef, pssm_bias, \
                pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
                [pdb_dict], cached_device, chain_dict, fixed_positions_dict, 
                omit_AA_dict=None, tied_positions_dict=None, pssm_dict=None, 
                bias_by_res_dict=None, ca_only=False
            )

            # Move tensors to device
            X = X.to(cached_device)
            S = S.to(cached_device)
            mask = mask.to(cached_device)
            chain_M = chain_M.to(cached_device)
            chain_M_pos = chain_M_pos.to(cached_device)
            residue_idx = residue_idx.to(cached_device)
            chain_encoding_all = chain_encoding_all.to(cached_device)

            # randn = torch.randn_like(S.float())
            randn = torch.randn(chain_M.shape, device=X.device)

            # Prepare node and edge embeddings
            from protein_mpnn_utils import gather_nodes, cat_neighbors_nodes

            E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)
            h_V_enc = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=cached_device)
            h_E = model.W_e(E)

            # Encoder is unmasked self-attention
            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            for i, layer in enumerate(model.encoder_layers):
                h_V_enc, h_E = layer(h_V_enc, h_E, E_idx, mask, mask_attend)

            # Concatenate sequence embeddings for autoregressive decoder
            h_S = model.W_s(S)
            h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

            # Build encoder embeddings
            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

            chain_M = chain_M*chain_M_pos*mask

            chain_M_np = chain_M.cpu().numpy()
            idx_to_loop = np.argwhere(chain_M_np[0,:]==1)[:,0]

            backbone_only = (mode == "conditional_probs_only_backbone")
            embeddings_dict_dt = {}
            for idx in idx_to_loop:
                h_V = torch.clone(h_V_enc)
                order_mask = torch.zeros(chain_M.shape[1], device=cached_device).float()
                if backbone_only:
                    order_mask = torch.ones(chain_M.shape[1], device=cached_device).float()
                    order_mask[idx] = 0.
                else:
                    order_mask = torch.zeros(chain_M.shape[1], device=cached_device).float()
                    order_mask[idx] = 1.
                decoding_order = torch.argsort((order_mask[None,]+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
                mask_size = E_idx.shape[1]
                permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
                order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=cached_device))), permutation_matrix_reverse, permutation_matrix_reverse)
                mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
                mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
                mask_bw = mask_1D * mask_attend
                mask_fw = mask_1D * (1. - mask_attend)

                h_EXV_encoder_fw = mask_fw * h_EXV_encoder
                decoder_layer_l = []
                for i, layer in enumerate(model.decoder_layers):
                    # Masked positions attend to encoder information, unmasked see. 
                    h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                    h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                    h_V = layer(h_V, h_ESV, mask)
                    decoder_layer_l.append(h_V[0, idx])
                
                sequence_embedding = h_S[0, idx].cpu()
                decoder_layer = torch.cat(decoder_layer_l, dim=-1)
                logits = model.W_out(h_V)
                log_probs = F.log_softmax(logits, dim=-1)
                log_conditional_probs = log_probs[:,idx,:]

                # Build return dictionary
                embeddings_dict = {
                    "sequence_embedding": sequence_embedding.squeeze(),  # [num_design, 128]
                    "decoder_layer": decoder_layer.squeeze().cpu(),  # [num_design, 128*3]
                    "conditional_probs": log_conditional_probs.squeeze().cpu(),  # [num_design, 21]
                }
                embeddings_dict_dt[idx+1] = embeddings_dict

        # Clean up temp directory
        shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, time.time() - start_time, {}

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"  Done ({elapsed_time:.2f}s)")

    return True, elapsed_time, embeddings_dict_dt


# ============== Example ==============
if __name__ == "__main__":
    # Test with example PDB
    pdb_path = "/home/DATA_2/hujiameng/tool_module/feature_extraction/ptmpnn/ProteinMPNN/inputs/1crn.pdb"

    success, elapsed, embeddings = get_embeddings(
        pdb_path=pdb_path,
        design_positions=None,  # All positions designable
        mode="conditional_probs_only"
    )

    print(f"Success: {success}, Time: {elapsed:.2f}s")
    if embeddings:
        # Print info for first position
        first_pos = list(embeddings.keys())[0]
        print(f"  Number of positions: {len(embeddings)}")
        print(f"  sequence_embedding shape: {embeddings[first_pos]['sequence_embedding'].shape}")
        print(f"  decoder_layer shape: {embeddings[first_pos]['decoder_layer'].shape}")
        print(f"  conditional_probs shape: {embeddings[first_pos]['conditional_probs'].shape}")
