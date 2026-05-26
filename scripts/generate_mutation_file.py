import os
import argparse
from typing import Dict, List, Optional, Tuple

import pandas as pd
from Bio.PDB import PDBParser, PDBIO
from Bio.Data.IUPACData import protein_letters_3to1



# ============== Configuration ==============
AA_LIST: List[str] = list("ACDEFGHIKLMNPQRSTVWY")


def _get_valid_residues(chain) -> Dict[int, str]:
    """
    Extract valid amino acid residues from a protein chain
    
    Args:
        chain: Bio.PDB chain object
        
    Returns:
        Dictionary mapping residue ID to single-letter amino acid code
        
    Raises:
        ValueError: If non-standard amino acid is found
    """
    residues: Dict[int, str] = {}
    
    for res in chain:
        resname = res.resname.capitalize()
        
        # Check if it's a standard amino acid
        if resname not in protein_letters_3to1:
            raise ValueError(f"Non-standard amino acid found: {res.resname}")
        
        res_id = res.id[1]
        residues[res_id] = protein_letters_3to1[resname]
    
    return residues


def get_mutation_file(
    pdb_file: str,
    seq_id: str,
    chain_id: str = "A",
    pos_limit = None
) -> Tuple[pd.DataFrame, str]:
    """
    Generate DMS mutations for all possible single-point mutations in a protein
    
    Generates all 19 possible amino acid substitutions for each residue position,
    creating a complete DMS dataset with both wild-type and mutant sequences.
    
    Args:
        pdb_file: Path to PDB structure file
        seq_id: Sequence identifier (e.g., UniProt ID)
        chain_id: Chain identifier (default: "A")
        pos_limit: Optional list of position IDs to include (default: None, all positions)
        
    Returns:
        Tuple of (mutation_dataframe: pd.DataFrame, message: str)
        DataFrame columns:
            - seq_id: Sequence identifier
            - chain: Chain identifier
            - position: Position in sequence (1-indexed)
            - wtAA: Wild-type amino acid
            - mutAA: Mutant amino acid
            - wt_seq: Complete wild-type sequence
            - mut_seq: Complete mutant sequence
            - pdb_file: Path to PDB file
            - pdb_position: Original PDB position (if different from sequence position)
        
    Raises:
        ValueError: If pdb_file does not exist or chain_id not found
        
    Example:
        >>> df, msg = get_mutation_file(
        ...     pdb_file="protein.pdb",
        ...     seq_id="P00533",
        ...     chain_id="A"
        ... )
        >>> print(f"Generated {len(df)} mutations")
        >>> print(df.head())
    """
    # Input validation
    pdb_file = os.path.abspath(pdb_file)
    
    if not os.path.exists(pdb_file):
        raise ValueError(f"PDB file not found: {pdb_file}")
    
    if not isinstance(seq_id, str) or not seq_id.strip():
        raise ValueError(f"Invalid seq_id: {seq_id}")
    
    if not isinstance(chain_id, str) or not chain_id.strip():
        raise ValueError(f"Invalid chain_id: {chain_id}")
        
    try:
        # Parse PDB structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(seq_id, pdb_file)
        model = structure[0]
        
        # Check chain exists
        if chain_id not in model:
            available_chains = list(model.child_dict.keys())
            raise ValueError(
                f"Chain {chain_id} not found in PDB file. "
                f"Available chains: {available_chains}"
            )
        
        chain = model[chain_id]
        res_dict = _get_valid_residues(chain)
        pdb_pos_l = list(res_dict.keys())

        # Filter by position limit if provided
        if pos_limit is not None:
            if not isinstance(pos_limit, list):
                pos_limit_split = pos_limit.split("-")
                pos_limit_l = list(range(int(pos_limit_split[0]), int(pos_limit_split[1])+1))
            else:
                pos_limit_l = pos_limit
        
        if not res_dict:
            raise ValueError("No valid residues found after filtering")
        
        # Generate all possible mutations
        position_list: List[int] = []
        wtAA_list: List[str] = []
        mutAA_list: List[str] = []
        
        for res_id, wtAA in res_dict.items():
            for mutAA in AA_LIST:
                if mutAA == wtAA:
                    continue
                position_list.append(int(res_id))
                wtAA_list.append(wtAA)
                mutAA_list.append(mutAA)
        
        # Create dataframe
        dms_df = pd.DataFrame({
            "position": position_list,
            "wtAA": wtAA_list,
            "mutAA": mutAA_list
        })
        
        dms_df['seq_id'] = seq_id
        dms_df['chain'] = chain_id
        dms_df['wt_seq'] = "".join(res_dict.values())
        dms_df['pdb_file'] = pdb_file
        col_order = ["seq_id", "chain", "position", "wtAA", "mutAA", "wt_seq", "mut_seq", "pdb_file"]
        dms_df['mut_seq'] = dms_df.apply(
                lambda row: row['wt_seq'][:row['position']-1] + row['mutAA'] + row['wt_seq'][row['position']:],
                axis=1
            )
        if pos_limit is not None:
            dms_df = dms_df[dms_df['position'].isin(pos_limit_l)]
        
        # Reorder columns
        dms_df = dms_df.loc[:, col_order]
        
        message = f"Generated {len(dms_df)} mutations for {len(res_dict)} positions"
        print(message)
        return dms_df
        
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error processing PDB file: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate saturation mutagenesis input file "
            "for STAB-DKL."
        )
    )

    parser.add_argument(
        "--pdb_file",
        type=str,
        required=True,
        help="Path to input PDB file."
    )

    parser.add_argument(
        "--seq_id",
        type=str,
        required=True,
        help=(
            "Sequence identifier used in output. "
        )
    )

    parser.add_argument(
        "--chain",
        type=str,
        required=True,
        help="Chain identifier."
    )

    parser.add_argument(
        "--pos_limit",
        type=str,
        default=None,
        help=(
            "Specific residue positions to mutate, eg. 10-20. "
            "If omitted, all positions are included."
        )
    )

    parser.add_argument(
        "--out_file",
        type=str,
        required=True,
        help=(
            "Output CSV file path. "
            "Default: <seq_id>_mutation_input.csv"
        )
    )

    return parser.parse_args()


def main():
    args = parse_args()

    out_file = args.out_file
    
    dms_df = get_mutation_file(
        pdb_file=args.pdb_file,
        seq_id=args.seq_id,
        chain_id=args.chain,
        pos_limit=args.pos_limit
    )

    out_dir = os.path.dirname(
        os.path.abspath(out_file)
    )

    os.makedirs(out_dir, exist_ok=True)

    dms_df.to_csv(out_file, index=False)

    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()