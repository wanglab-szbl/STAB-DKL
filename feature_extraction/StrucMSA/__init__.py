"""
Title: StrucMSA Feature Extraction
Author: gumingmu
Date: 2025-07-30
Description: Extract structure-based MSA features for protein stability prediction.
"""
# pylint: disable=invalid-name

import os
from collections import Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from Bio import BiopythonDeprecationWarning
from Bio.PDB import PDBParser, NeighborSearch
from Bio.SeqUtils import seq1
from joblib import Parallel, delayed
from tqdm import tqdm

# Constants
DEFAULT_NEIBOR_PERCENT_L: List[float] = [0.5, 0.75, 1.0]
DEFAULT_RADIUS_L: List[int] = list(range(1, 21, 1))
DEFAULT_AA_LIST: str = "ACDEFGHIKLMNPQRSTVWY"


def get_neibor_residues(df: pd.DataFrame, 
                        radius_l: List[int] = None) -> Dict[str, Any]:
    """
    Calculate neighbor residues for each position in the protein structure.

    Args:
        df: DataFrame with columns ['seq_id', 'chain', 'position', 'pdb_file']
        radius_l: List of radius values to search (default: 1-20)

    Returns:
        Nested dictionary with neighbor information
    """
    if radius_l is None:
        radius_l = DEFAULT_RADIUS_L
    
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    required_cols = ['seq_id', 'chain', 'position', 'pdb_file']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    parser = PDBParser(QUIET=True)
    uni_seq_id = list(set(df['seq_id'].tolist()))
    neibor_dt = {}

    for seq_id in tqdm(uni_seq_id, total=len(uni_seq_id), desc="calculate neibor resi"):
        neibor_dt[seq_id] = {}
        sub_df = df[df['seq_id'] == seq_id]
        pdb_file = sub_df['pdb_file'].tolist()[0]
        
        structure = parser.get_structure("prot", pdb_file)
        model = structure[0]
        chain = sub_df['chain'].tolist()[0]
        model_chain = model[chain]
        atoms = [atom for atom in model_chain.get_atoms()]
        ns = NeighborSearch(atoms)
        
        uni_position = list(set(sub_df['position'].tolist()))
        for position in uni_position:
            neibor_dt[seq_id][str(position)] = {}
            target_res = model_chain[position]
            
            for radius in radius_l:
                close_atoms = ns.search(target_res["CA"].coord, radius, level="R")
                pos, res_l = [], []
                for res in close_atoms:
                    if res.get_id()[1] != position:
                        pos.append(int(res.get_id()[1]))
                        res_l.append(seq1(res.get_resname()))
                neibor_dt[seq_id][str(position)][str(radius)] = [pos, res_l]
    
    return neibor_dt


def parser_msa(msa_file: str) -> Dict[str, List[str]]:
    """
    Parse a3m format MSA file containing multiple sequences.
    The file is separated by \x00 character between different sequence MSAs.

    Args:
        msa_file: Path to the a3m format MSA file

    Returns:
        Dictionary with sequence ID as key and list of MSA sequences as value

    Raises:
        FileNotFoundError: If the MSA file does not exist
        ValueError: If MSA sequences have inconsistent lengths
    """
    if not os.path.exists(msa_file):
        raise FileNotFoundError(f"MSA file not found: {msa_file}")

    with open(msa_file, 'rb') as f:
        content = f.read()

    # Split by \x00 separator to get individual MSA blocks
    msa_blocks = content.split(b"\x00")
    result = {}

    for block in msa_blocks:
        if not block.strip():
            continue

        try:
            block_str = block.decode('utf-8')
            lines = block_str.strip().split('\n')

            if not lines or not lines[0].startswith('>'):
                continue

            header_line = lines[0]
            seq_id = header_line[1:].strip()

            # Parse MSA sequences (alternating header and sequence lines)
            msa_sequences = []
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('>'):
                    if i + 1 < len(lines):
                        seq_line = lines[i + 1].strip()
                        # Remove lowercase letters (deletion markers in a3m)
                        seq_clean = "".join([c for c in seq_line if not c.islower()])
                        msa_sequences.append(seq_clean)
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1

            # Check if all MSA sequences have consistent lengths
            if msa_sequences:
                seq_lengths = [len(seq) for seq in msa_sequences]
                if len(set(seq_lengths)) > 1:
                    query_len = len(msa_sequences[0])
                    raise ValueError(
                        f"Inconsistent MSA sequence lengths for {seq_id}: "
                        f"query sequence length={query_len}, "
                        f"found inconsistent lengths {set(seq_lengths)}"
                    )
                result[seq_id] = msa_sequences

        except UnicodeDecodeError:
            print(f"Warning: Failed to decode MSA block")
            continue

    return result


def get_freq(task: Dict[str, Any], 
             msa_dt: Dict[str, List[str]], 
             neibor_dt: Dict[str, Any], 
             aa_list: str = DEFAULT_AA_LIST) -> Tuple[str, int, List[List[float]]]:
    """
    Calculate frequency of amino acids at a given position.

    Args:
        task: Dict with 'seq_id' and 'position'
        msa_dt: Dictionary of MSA sequences
        neibor_dt: Dictionary of neighbor residues
        aa_list: String of amino acid letters

    Returns:
        Tuple of (seq_id, position, freq_l) where freq_l is list of 61 feature vectors
    """
    seq_id = task['seq_id']
    position = int(task['position'])
    msa_l = msa_dt[seq_id]
    n_seqs = len(msa_l)
    pos_idx = position - 1
    freq_l = []

    # Pre-extract amino acids at target position for all sequences
    aa_at_pos = [seq[pos_idx] for seq in msa_l]

    # Full sequence frequency
    counter = Counter(aa_at_pos)
    full_sq_out = [counter.get(aa, 0) / n_seqs for aa in aa_list]
    freq_l.append(full_sq_out)

    # Neighbor-based frequencies - precompute neighbor match scores
    for radius in DEFAULT_RADIUS_L:
        neibor_l = neibor_dt[seq_id][str(position)][str(radius)]
        n_neighbors = len(neibor_l[0])

        if n_neighbors == 0:
            # No neighbors - all sequences pass with same frequency
            for _ in DEFAULT_NEIBOR_PERCENT_L:
                freq_l.append(full_sq_out)
            continue

        # Precompute match scores for all sequences
        neighbor_positions = [int(x) - 1 for x in neibor_l[0]]
        neighbor_aas = neibor_l[1]

        match_scores = []
        for seq in msa_l:
            matches = sum(1 for x, y in zip(neighbor_positions, neighbor_aas) if seq[x] == y)
            match_scores.append(matches / n_neighbors)

        # Use precomputed scores for each neibor_percent
        for neibor_percent in DEFAULT_NEIBOR_PERCENT_L:
            filtered_aa = [aa for aa, score in zip(aa_at_pos, match_scores) if score >= neibor_percent]
            counter = Counter(filtered_aa)
            freq_l.append([counter.get(aa, 0) / len(filtered_aa) for aa in aa_list])

    return seq_id, position, freq_l


def _process_seq_group(seq_id: str, 
                       positions: List[int], 
                       msa_l: List[str], 
                       neibor_dt_sub: Dict[str, Any], 
                       aa_list: str) -> List[Tuple[str, int, List[List[float]]]]:
    """
    Process all positions for a single seq_id.

    Args:
        seq_id: Sequence ID
        positions: List of positions to process
        msa_l: List of MSA sequences
        neibor_dt_sub: Subset of neighbor dictionary for this seq_id
        aa_list: String of amino acid letters

    Returns:
        List of results for each position
    """
    results = []
    for position in positions:
        task = {'seq_id': seq_id, 'position': position}
        result = get_freq(task, {seq_id: msa_l}, neibor_dt_sub, aa_list)
        results.append(result)
    return results


def get_strucMSA(df: pd.DataFrame, 
                 msa_file: str, 
                 n_jobs: int, 
                 aa_list: str = DEFAULT_AA_LIST) -> None:
    """
    Calculate structure-based MSA features and save as PyTorch format.

    Args:
        df: Input DataFrame with seq_id, chain, position, pdb_file
        msa_file: Path to MSA file
        n_jobs: Number of parallel jobs
        aa_list: Amino acid letters
    """
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive integer, got {n_jobs}")

    # Parse MSA
    msa_dt = parser_msa(msa_file)

    # Get neighbor residues
    neibor_dt = get_neibor_residues(df)

    # Group by seq_id for better parallel efficiency
    df_unique = df[['seq_id', 'position']].drop_duplicates()
    seq_groups = df_unique.groupby('seq_id')['position'].apply(list).to_dict()

    # Prepare tasks - each task gets only its needed data
    tasks = [
        (seq_id, positions, msa_dt[seq_id], {seq_id: neibor_dt[seq_id]}, aa_list)
        for seq_id, positions in seq_groups.items()
    ]

    # Adjust n_jobs if task count is small
    actual_n_jobs = min(n_jobs, len(tasks))

    # Parallel processing by seq_id
    batch_results = Parallel(n_jobs=actual_n_jobs, backend='loky')(
        delayed(_process_seq_group)(seq_id, positions, msa_l, neibor_dt_sub, aa_list)
        for seq_id, positions, msa_l, neibor_dt_sub, aa_list in tqdm(tasks, total=len(tasks))
    )

    # Flatten results
    results = [item for batch in batch_results for item in batch]

    # Build feature matrix: [n_samples, 61, 20]
    n_samples = len(results)
    n_features = 61  # 1 (full) + 20 (radius) * 3 (neibor_percent)
    n_aa = len(aa_list)

    features = np.zeros((n_samples, n_aa, n_features))
    dim0_data = []

    for idx, (seq_id, position, freq_l) in enumerate(results):
        dim0_data.append({'seq_id': seq_id, 'position': int(position)})
        for feat_idx, freq in enumerate(freq_l):
            features[idx, :, feat_idx] = freq

    # Build dim1 labels
    dim2_labels = ['strucMSA_full']
    for radius in DEFAULT_RADIUS_L:
        for neibor_percent in DEFAULT_NEIBOR_PERCENT_L:
            dim2_labels.append(f"strucMSA_r{radius}_p{neibor_percent}")

    # Create output dictionary in PyTorch format
    out_dt = {
        'features': torch.tensor(features, dtype = torch.float32),
        'dim0': pd.DataFrame(dim0_data),
        'dim1': aa_list,
        'dim2': dim2_labels
    }

    return out_dt
