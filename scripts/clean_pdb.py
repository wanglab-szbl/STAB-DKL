from typing import Dict

import argparse
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO
from Bio.Data.IUPACData import protein_letters_3to1



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
            continue
                
        res_id = res.id[1]
        residues[res_id] = protein_letters_3to1[resname]
    
    return residues


def clean_pdb(
    input_pdb: str,
    out_pdb: str,
    chain_id: str
):
    """
    Renumber residues in a PDB file.

    This function:
    - Renumbers residues to start from 1
    - Converts discontinuous residue numbering
      into continuous numbering
    """
    # Parse PDB structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("a", input_pdb)
    model = structure[0]
    
    # Check chain exists
    if chain_id not in model:
        available_chains = list(model.child_dict.keys())
        raise ValueError(
            f"Chain {chain_id} not found in PDB file. "
            f"Available chains: {available_chains}"
        )
    
    # clean chain
    for chain_obj in list(model):
        if chain_obj.id != chain_id:
            model.detach_child(chain_obj.id)
            print(f"Remove chain {chain_obj.id}")
    chain = model[chain_id]
    res_dict = _get_valid_residues(chain)
    pdb_pos_l = list(res_dict.keys())
    # clean residue
    for res in list(chain):
        if not res.id[1] in pdb_pos_l:
            chain.detach_child(res.id)
            print(f"Remove unsupported residues {res.resname}")
    
    pdb_pos_check = (min(pdb_pos_l) == 1 and max(pdb_pos_l) == len(pdb_pos_l))
    if not pdb_pos_check:
        pdb_pos_l_rank = [sorted(pdb_pos_l).index(x) + 1 for x in pdb_pos_l]
        pdb_pos_map = dict(zip(pdb_pos_l, pdb_pos_l_rank))
        for res in chain:
            res.id = (res.id[0], pdb_pos_map[res.id[1]], res.id[2])
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Renumber PDB residues to make residue "
            "indices continuous and start from 1."
        )
    )

    parser.add_argument(
        "--input_pdb",
        type=str,
        required=True,
        help="Input PDB file"
    )

    parser.add_argument(
        "--chain_id",
        type=str,
        required=True,
        help="Chain ID to process (e.g. A)"
    )

    parser.add_argument(
        "--out_pdb",
        type=str,
        required=True,
        help="Output PDB file "
    )

    args = parser.parse_args()

    input_pdb = Path(args.input_pdb)
    out_pdb = Path(args.out_pdb)

    out_pdb.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    clean_pdb(
        input_pdb=str(input_pdb),
        out_pdb=str(out_pdb),
        chain_id=args.chain_id
    )

    print(f"Fixed PDB saved to: {out_pdb}")
    

if __name__ == "__main__":
    main()
    