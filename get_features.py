"""Get features (ESM2, proteinMPNN, strucMSA) for all datasets.

Usage:
    python -u get_features.py --config config/features.yaml
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/feature_extraction")

from ESM2 import clear_cache as clear_esm2_cache
from ESM2 import get_embeddings as get_esm2_emb
from StrucMSA import get_strucMSA
from mmseqs2 import get_msa_cpu, get_msa_gpu
from ptmpnn.mutation_features import clear_cache as clear_ptmpnn_cache
from ptmpnn.mutation_features import get_embeddings as get_ptmpnn_emb


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration dictionary.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required fields are missing
    """
    required_fields = ['dataset_files']
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    if not isinstance(config['dataset_files'], list):
        raise ValueError("dataset_files must be a list")


def generate_msa(seq_list: List[Dict[str, str]], msa_dir: str, mmseqs_cmd: str,
                 mmseqs_db: str, use_gpu: bool = True, max_seqs: int = 1000,
                 threads: int = 24) -> None:
    """Generate MSA for all sequences using mmseqs2.

    Args:
        seq_list: List of dicts with id, sequence
        msa_dir: MSA output directory
        mmseqs_cmd: Path to mmseqs2 executable
        mmseqs_db: Target database for alignment
        use_gpu: Use GPU version
        max_seqs: Maximum number of sequences
        threads: Number of threads
    """
    sequences = [{"id": s["id"], "sequence": s["sequence"]} for s in seq_list]
    os.makedirs(msa_dir, exist_ok=True)
    print(f"Generating MSA for {len(sequences)} sequences...")

    msa_func = get_msa_gpu if use_gpu else get_msa_cpu
    success, _ = msa_func(
        sequences=sequences,
        outdir=msa_dir,
        targetDB=mmseqs_db,
        max_seqs=max_seqs,
        threads=threads,
        mmseqs_path=mmseqs_cmd
    )

    if not success:
        raise RuntimeError("MSA generation failed")


def get_strucmsa_features(df: pd.DataFrame, msa_file: str, output_prefix: str,
                          mmseqs_config: Dict[str, Any], n_jobs: int) -> None:
    """Generate or load MSA and extract StrucMSA features.

    Args:
        df: Input DataFrame with mutation data
        msa_file: Path to MSA file
        output_prefix: Output file prefix
        mmseqs_config: MSA generation configuration
        n_jobs: Number of parallel jobs
    """
    if os.path.exists(msa_file):
        print(f"\nUsing provided MSA file: {msa_file}")
    else:
        print(f"\nGenerating MSA with mmseqs2...")
        unique_df = df[['seq_id', 'wt_seq']].drop_duplicates()
        seq_list = [{"id": row['seq_id'], "sequence": row['wt_seq']}
                    for _, row in unique_df.iterrows()]
        print(f"Unique sequences: {len(seq_list)}")
        generate_msa(seq_list, os.path.dirname(msa_file), **mmseqs_config)

    print("\nExtracting StrucMSA features...")
    output_path = output_prefix + "strucmsa.pt"
    out_dt = get_strucMSA(df, msa_file, n_jobs)
    torch.save(out_dt, output_path)
    print(f"\nDone! strucMSA features saved to: {output_path}")


def get_esm2_features(df: pd.DataFrame, esm2_path: str, output_prefix: str) -> None:
    """Generate ESM2 features.

    Args:
        df: Input DataFrame with mutation data
        esm2_path: Path to ESM2 model
        output_prefix: Output file prefix
    """
    print(f"\nGenerating ESM2 features...")

    # Prepare data: combine wt and mut sequences
    df1 = df.copy().rename(columns={"wt_seq": "seq"})
    df2 = df.copy().rename(columns={"mut_seq": "seq"})
    df1['id'] = df1['seq_id'] + "_" + df1['position'].astype(str) + "_" + df1['wtAA']
    df2['id'] = df2['seq_id'] + "_" + df2['position'].astype(str) + "_" + df2['mutAA']

    all_col = ['id', 'seq', 'position']
    df = pd.concat([df1[all_col], df2[all_col]], ignore_index=True).drop_duplicates()
    df_dt_l = df.groupby("id").agg({"seq": "first", "position": list}).reset_index().to_dict(orient="records")

    # Get features
    df_dt = df_dt_l[0]
    result = get_esm2_emb(df_dt, feature_types=["pos"], half_precision=False,
                          model_path=esm2_path, verbose=False)
    features = torch.empty((df.shape[0], result['emb']['pos'][df_dt['position'][0]].shape[0]))

    id_l, position_l = [], []
    for i, df_dt in enumerate(tqdm(df_dt_l, total=len(df_dt_l))):
        result = get_esm2_emb(df_dt, feature_types=["pos"], half_precision=False,
                              model_path=esm2_path, verbose=False)
        for position, feature in result['emb']['pos'].items():
            features[i, :] = feature
            id_l.append(result['id'])
            position_l.append(position)

    out_dt = {
        "features": features,
        "dim0": pd.DataFrame({"id": id_l, "position": position_l}),
        "dim1": ["ESM2_" + str(i) for i in range(features.shape[1])]
    }
    output_path = output_prefix + "esm2.pt"
    torch.save(out_dt, output_path)
    clear_esm2_cache()
    print(f"\nDone! ESM2 features saved to: {output_path}")


def get_ptmpnn_features(df: pd.DataFrame, output_prefix: str) -> None:
    """Generate ProteinMPNN features.

    Args:
        df: Input DataFrame with mutation data
        output_prefix: Output file prefix
    """
    print(f"\nGenerating ProteinMPNN features...")

    df = df[['seq_id', 'pdb_file', 'position']].drop_duplicates()
    df_dt_l = df.groupby("seq_id").agg({"pdb_file": "first", "position": list}).reset_index().to_dict(orient="records")

    features = torch.empty((df.shape[0], 1024 + 42))
    seq_id_l, position_l = [], []
    i = 0

    for df_dt in tqdm(df_dt_l, total=len(df_dt_l), desc="get_ptmpnn_features"):
        result_l = []
        for mode in ["conditional_probs_only", "conditional_probs_only_backbone"]:
            success, elapsed, result = get_ptmpnn_emb(
                pdb_path=df_dt['pdb_file'],
                design_positions=df_dt['position'],
                verbose=False,
                mode=mode
            )
            result_l.append(result)

        result1, result2 = result_l[0], result_l[1]
        for position in list(result1.keys()):
            feature = torch.cat(list(result1[position].values()) + list(result2[position].values()), dim=0)
            features[i, :] = feature
            seq_id_l.append(df_dt['seq_id'])
            position_l.append(position)
            i += 1

    dim0_df = pd.DataFrame({"seq_id": seq_id_l, "position": position_l})
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    dim1_l1 = ["seq_" + str(i) for i in range(128)] + \
              ["decoder_" + str(i) for i in range(128 * 3)] + \
              ["prob_" + _ for _ in alphabet]
    dim1_l = ["sqbb_" + _ for _ in dim1_l1] + ["bb_" + _ for _ in dim1_l1]

    out_dt = {
        "features": features,
        "dim0": dim0_df,
        "dim1": dim1_l
    }
    output_path = output_prefix + "ptmpnn.pt"
    torch.save(out_dt, output_path)
    clear_ptmpnn_cache()
    print(f"\nDone! ProteinMPNN features saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Get features for protein stability prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
        python -u get_features.py --config config/default.yaml
        python -u get_features.py --config config/custom.yaml
            """
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load and validate configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    validate_config(config)

    # Extract configuration values
    pdb_dir = config.get("pdb_dir")
    dataset_files = config['dataset_files']
    msa_file = config.get('msa_file', "features/msa/msa.fa")
    output_prefix = config.get('output_prefix', "data/features/")
    os.makedirs(output_prefix, exist_ok=True)

    # MSA generation settings
    msa_gen_config = config.get('msa_generation', {})
    use_gpu = msa_gen_config.get('use_gpu', True)
    max_seqs = msa_gen_config.get('max_seqs', 1000)
    threads = msa_gen_config.get('threads', 22)
    mmseqs_cmd = msa_gen_config.get('mmseqs_cmd', "mmseqs")
    mmseqs_db = msa_gen_config.get('mmseqs_db')

    # Feature extraction settings
    feat_config = config.get('feature_extraction', {})
    n_jobs = feat_config.get('n_jobs', 22)

    # ESM2 settings
    esm2_path = config.get('esm2_path')
    if esm2_path is None:
        raise ValueError("ESM2 path is not provided")

    # Chain settings
    force_chain_A = config.get('force_chain_A', False)

    # Validate MSA settings
    if not os.path.exists(msa_file) and mmseqs_db is None:
        raise ValueError("MSA file does not exist and mmseqs_db is not provided")

    # Print configuration summary
    print("=" * 60)
    print("Configuration:")
    print(f"  Input datasets: {dataset_files}")
    print(f"  PDB directory: {pdb_dir}")
    print(f"  MSA file: {msa_file}")
    print(f"  mmseqs cmd: {mmseqs_cmd}")
    print(f"  mmseqs db: {mmseqs_db}")
    print(f"  Use GPU: {use_gpu}")
    print(f"  Max sequences: {max_seqs}")
    print(f"  Threads: {threads}")
    print(f"  Output: {output_prefix}")
    print(f"  N jobs: {n_jobs}")
    print(f"  ESM2 path: {esm2_path}")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    all_dfs = []
    for csv_file in dataset_files:
        if not os.path.exists(csv_file):
            raise ValueError(f"Dataset file not found: {csv_file}")
        all_dfs.append(pd.read_csv(csv_file))

    if not all_dfs:
        raise ValueError("No data files loaded")

    df = pd.concat(all_dfs, ignore_index=True)

    # Process seq_id and chain
    if "seq_id" not in df.columns:
        df['seq_id'] = df.apply(lambda x: x['pdb'] + "_" + x['chain'], axis=1)
    if force_chain_A:
        df['chain'] = 'A'

    # Add pdb_file path
    if "pdb_file" not in df.columns and pdb_dir:
        df['pdb_file'] = df.apply(
            lambda x: os.path.join(pdb_dir, x['seq_id'].lower() + "_model.pdb"),
            axis=1
        )
    else:
        raise ValueError("pdb_file not in columns and pdb_dir is None")

    print(f"Total records: {len(df)}")

    # Get features
    mmseqs_config = {
        'mmseqs_cmd': mmseqs_cmd,
        'mmseqs_db': mmseqs_db,
        'use_gpu': use_gpu,
        'max_seqs': max_seqs,
        'threads': threads
    }

    print("\n[1/3] Extracting StrucMSA features...")
    get_strucmsa_features(df, msa_file, output_prefix, mmseqs_config, n_jobs)
    print("\n[2/3] Extracting ESM2 features...")
    get_esm2_features(df, esm2_path, output_prefix)
    torch.cuda.empty_cache()
    print("\n[3/3] Extracting PTMPNN features...")
    get_ptmpnn_features(df, output_prefix)
    print("\nFeature extraction completed!")


if __name__ == "__main__":
    main()
