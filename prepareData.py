"""
Data preparation script for stability prediction.
"""

import os
import time
from datetime import datetime
from copy import deepcopy

import requests
import pandas as pd
from tqdm import tqdm
from Bio import PDB
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1


# ============== Constants ==============
DATASETS = {
    "train": ["S11278", "S5052"],
    "test": ["S669", "p53", "myoglobin", "ssym"]
}
REMOVE_PDB = ["1SEE"]  # Non-experimental structures
MOD_MAPPING = {
    "FTR": "TRP", "PCA": "GLN", "ABA": "CYS",
    "DSN": "SER", "KCX": "LYS"
}
COL_SEL = ['pdb', 'chain', 'position', 'wtAA', 'mutAA', 'wt_seq', 'mut_seq', 'label']


# ============== PDB Download ==============
def download_pdb(pdb_id: str, out_dir: str, max_retries: int = 3, delay: int = 5) -> str:
    """Download PDB structure in CIF format."""
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    out_path = os.path.join(out_dir, f"{pdb_id}.cif")

    if os.path.exists(out_path):
        return out_path

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}: Downloading {pdb_id}...")
            response = requests.get(url, timeout=100)
            if response.status_code == 200:
                with open(out_path, "w") as f:
                    f.write(response.text)
                print(f"Downloaded {pdb_id} successfully.")
                return out_path
        except requests.RequestException as e:
            print(f"Request failed: {e}")

        if attempt < max_retries:
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)

    raise Exception(f"Failed to download {pdb_id} after {max_retries} attempts.")


# ============== Sequence Extraction ==============
def get_wt_seq(cif_dict: dict, chain: str) -> dict:
    """Extract wild-type sequence from CIF structure."""
    seq_id = cif_dict['_pdbx_poly_seq_scheme.pdb_seq_num']
    seq_id1 = cif_dict['_pdbx_poly_seq_scheme.pdb_ins_code']
    pdb_strand_id = cif_dict['_pdbx_poly_seq_scheme.pdb_strand_id']
    mon_id = cif_dict['_pdbx_poly_seq_scheme.mon_id']

    # Remove non-polymer residues
    remove_aa = [
        cif_dict['_chem_comp.id'][i]
        for i in range(len(cif_dict['_chem_comp.id']))
        if cif_dict['_chem_comp.type'][i] == "non-polymer"
    ]

    seq_l = [
        mon_id[i] for i in range(len(mon_id))
        if pdb_strand_id[i] == chain and mon_id[i] not in remove_aa
    ]

    # Handle insertion codes
    seq_id = [
        seq_id[i] if seq_id1[i] == "." else seq_id[i] + seq_id1[i]
        for i in range(len(seq_id))
    ]

    # Handle modified residues
    if '_pdbx_modification_feature.label_comp_id' in cif_dict.keys():
        mod_name = cif_dict['_pdbx_modification_feature.label_comp_id']
        mod_ori_name = cif_dict['_pdbx_modification_feature.modified_residue_id']
        mod_name = [mod_name[i] for i in range(len(mod_name)) if len(mod_ori_name[i]) == 3]
        mod_ori_name = [mod_ori_name[i] for i in range(len(mod_name)) if len(mod_ori_name[i]) == 3]
        if len(mod_name) != 0:
            mod_mapping = dict(zip(mod_name, mod_ori_name))
            seq_l = [mod_mapping.get(_, _) for _ in seq_l]

    # Apply custom mapping
    seq_l = [MOD_MAPPING.get(_, _) for _ in seq_l]

    ss_l = [
        seq_id[i] for i in range(len(seq_id))
        if pdb_strand_id[i] == chain and mon_id[i] not in remove_aa
    ]

    # Check for non-standard amino acids
    aa_ns = [_ for _ in seq_l if not PDB.is_aa(_, standard=True)]
    if aa_ns:
        print(f"Found non-standard residues: {aa_ns}")
        exit()

    seq_l = [seq1(_) for _ in seq_l]
    return dict(zip(ss_l, seq_l))


def get_chain_mapping(cif_dict: dict, chain: str) -> str:
    """Get auth chain ID from label chain ID."""
    auth_chain_ids = cif_dict["_atom_site.auth_asym_id"]
    label_chain_ids = cif_dict["_atom_site.label_asym_id"]
    mapping = {}
    for auth, label in zip(auth_chain_ids, label_chain_ids):
        if label not in mapping:
            mapping[label] = auth
    return mapping.get(chain, chain)


# ============== Data Loading ==============
def get_seq(pdb_dir: str = "data/pdb/") -> pd.DataFrame:
    """Download PDB structures and extract sequences."""
    os.makedirs(pdb_dir, exist_ok=True)

    data_df_l = []
    for dataset_type, dataset_list in DATASETS.items():
        for dataset_name in dataset_list:
            input_file = f"data/meta/{dataset_name}.csv"

            data_df = pd.read_csv(input_file)
            data_df = data_df[~data_df['pdb'].isin(REMOVE_PDB)]
            data_df.sort_values(by="pdb", inplace=True)
            data_df = data_df[~(data_df['chain'] == "unsigned")]

            # Download PDBs
            uni_pdb = data_df["pdb"].drop_duplicates().tolist()
            print(f"[{datetime.now()}] Downloading PDBs for {dataset_name}")
            for pdb_id in tqdm(uni_pdb, desc=f"Download {dataset_name}"):
                download_pdb(pdb_id, pdb_dir)

            # Extract sequences (with caching)
            print(f"[{datetime.now()}] Extracting sequences for {dataset_name}")

            # Cache for CIF dicts and sequences
            cif_cache = {}
            seq_cache = {}

            # Pre-compute unique pdb+chain combinations
            pdb_chain_df = data_df[['pdb', 'chain']].drop_duplicates()

            for _, rc in tqdm(pdb_chain_df.iterrows(), total=len(pdb_chain_df), desc=f"Cache sequences"):
                pdb_id = rc['pdb']
                chain = rc['chain']

                # Load CIF if not cached
                if pdb_id not in cif_cache:
                    pdb_path = os.path.join(pdb_dir, f"{pdb_id.lower()}.cif")
                    cif_cache[pdb_id] = MMCIF2Dict(pdb_path)

                cif_dict = cif_cache[pdb_id]

                # Handle chain mapping
                mapped_chain = chain
                if chain not in list(set(cif_dict["_atom_site.auth_asym_id"])):
                    mapped_chain = get_chain_mapping(cif_dict, chain)
                    if mapped_chain not in list(set(cif_dict["_atom_site.auth_asym_id"])):
                        seq_cache[(pdb_id, chain)] = None
                        continue

                # Extract sequence
                wt_seq = get_wt_seq(cif_dict, mapped_chain)
                seq_cache[(pdb_id, chain)] = wt_seq

            # Process rows using cache
            wt_seqs, mut_seqs = [], []
            for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc=f"Process {dataset_name}"):
                pdb_id = row['pdb']
                chain = row['chain']
                wtAA = row['wtAA']
                mutPos = str(row['mutPos'])
                mutAA = row['mutAA']

                # Get from cache
                wt_seq_dt = seq_cache.get((pdb_id, chain))

                if wt_seq_dt is None:
                    wt_seqs.append("None")
                    mut_seqs.append("None")
                    continue

                mut_seq_dt = deepcopy(wt_seq_dt)

                if mutPos in mut_seq_dt.keys():
                    if wtAA == mut_seq_dt[mutPos]:
                        mut_seq_dt[mutPos] = mutAA
                        mut_seq = "".join(mut_seq_dt.values())
                        wt_seqs.append("".join(wt_seq_dt.values()))
                        mut_seqs.append(mut_seq)
                    else:
                        wt_seqs.append("None")
                        mut_seqs.append("None")
                else:
                    wt_seqs.append("None")
                    mut_seqs.append("None")

            data_df['mut_seq'] = mut_seqs
            data_df['wt_seq'] = wt_seqs
            data_df['position'] = data_df.apply(
                lambda x: next(
                    (i for i in range(len(x['wt_seq'])) if x['wt_seq'][i] != x['mut_seq'][i]),
                    -1
                ) + 1,
                axis=1
            )
            data_df.columns = data_df.columns.to_series().apply(
                lambda x: "label" if ("ddG" in x or "dTm" in x) else x
            )
            data_df = data_df[COL_SEL]
            data_df['dataset_name'] = dataset_name
            data_df['dataset_type'] = dataset_type
            data_df_l.append(data_df)

            # Statistics
            counts = data_df['wt_seq'].value_counts()
            print(f"{dataset_name}: None={counts.get('None', 0)}")

    data_df = pd.concat(data_df_l, axis=0)
    return data_df


def get_megascale() -> pd.DataFrame:
    """Load and process megascale dataset."""
    data_df = pd.read_csv("data/meta/ms.csv")
    data_df = data_df.loc[data_df['index'] >= 0]
    data_df = data_df.loc[~data_df['WT_name'].str.contains('_', regex=False)]
    data_df['pdb'] = data_df['name'].apply(lambda x: x.split(".")[0] + "-ms")
    data_df['chain'] = "A"
    data_df['wtAA'] = data_df.apply(lambda x: x['wt_seq'][x['index']], axis=1)
    data_df['mutAA'] = data_df.apply(lambda x: x['mut_seq'][x['index']], axis=1)
    data_df['position'] = data_df['index'].copy() + 1
    data_df['label'] = data_df['ddG']
    data_sub_df = data_df.loc[:, COL_SEL]
    data_sub_df['dataset_name'] = "ms"
    data_sub_df['dataset_type'] = "train"
    return data_sub_df


# ============== Sequence Identity Filtering ==============
def mmseqs_search(query_fa: str, target_fa: str, out_file: str,
                  cutoff: float = 30, cov_mode: int = 0) -> pd.DataFrame:
    """Run mmseqs search and filter by identity."""
    os.system(
        f"mmseqs easy-search {query_fa} {target_fa} {out_file} tmp "
        f"-e inf --cov-mode {cov_mode} --alignment-mode 3 "
        f"--format-output 'query,target,pident,alnlen,qlen,tlen,evalue'"
    )

    ident_df = pd.read_csv(out_file, sep='\t', header=None)
    ident_df.columns = ['query', 'target', 'pident', 'alnlen', 'qlen', 'tlen', 'evalue']
    ident_df['identity'] = ident_df['pident'] * ident_df['alnlen'] / ident_df[['qlen', 'tlen']].max(axis=1)

    return ident_df[ident_df['identity'] > cutoff]


def filter_by_identity(query_df: pd.DataFrame, target_df: pd.DataFrame,
                       cutoff: float = 30, seq_col: list = None) -> pd.DataFrame:
    """
    Filter query sequences that are too similar to target sequences.

    Args:
        query_df: DataFrame to filter
        target_df: DataFrame containing target sequences
        cutoff: Identity threshold (default 30%)
        seq_col: Column name for seqid and seq (unused, kept for compatibility)

    Returns:
        Filtered DataFrame
    """
    import tempfile

    if seq_col is None:
        seq_col = ['seq_id', 'wt_seq']

    # Create temp FASTA files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        query_fa = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        target_fa = f.name
    with tempfile.NamedTemporaryFile(suffix='', delete=False) as f:
        result_file = f.name

    try:
        # Create seq_id for query
        query_df = query_df.copy()
        query_df['seq_id'] = query_df.apply(lambda x: x['pdb'] + "_" + x['chain'], axis=1)
        query_fa_df = query_df[['seq_id', 'wt_seq']].drop_duplicates().reset_index(drop=True)
        with open(query_fa, 'w') as f:
            for _, row in query_fa_df.iterrows():
                f.write(f">{row['seq_id']}\n{row['wt_seq']}\n")

        # Create seq_id for target
        target_df = target_df.copy()
        target_df['seq_id'] = target_df.apply(lambda x: x['pdb'] + "_" + x['chain'], axis=1)
        target_fa_df = target_df[['seq_id', 'wt_seq']].drop_duplicates().reset_index(drop=True)
        with open(target_fa, 'w') as f:
            for _, row in target_fa_df.iterrows():
                f.write(f">{row['seq_id']}\n{row['wt_seq']}\n")

        # Run mmseqs
        print(f"Running mmseqs search (cutoff={cutoff}%)...")
        os.system(
            f"mmseqs easy-search {query_fa} {target_fa} {result_file} tmp "
            f"-e inf --cov-mode 0 --alignment-mode 3 "
            f"--format-output 'query,target,pident,alnlen,qlen,tlen' > /dev/null 2>&1"
        )

        # Parse results
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            ident_df = pd.read_csv(result_file, sep='\t', header=None)
            ident_df.columns = ['query', 'target', 'pident', 'alnlen', 'qlen', 'tlen']
            ident_df['identity'] = ident_df['pident'] * ident_df['alnlen'] / ident_df[['qlen', 'tlen']].max(axis=1)

            # Get sequences with identity > cutoff
            similar = set(ident_df[ident_df['identity'] > cutoff]['query'].unique())
            print(f"Found {len(similar)} sequences with identity > {cutoff}%")

            # Filter query_df: keep sequences NOT in similar set
            filtered_df = query_df[~query_df['seq_id'].isin(similar)].reset_index(drop=True)
            filtered_df = filtered_df.drop(columns=['seq_id'])

            print(f"Filtered: {len(query_df)} -> {len(filtered_df)} (removed {len(query_df) - len(filtered_df)})")
            return filtered_df
        else:
            print("No hits found, returning original dataframe")
            if 'seq_id' in query_df.columns:
                query_df = query_df.drop(columns=['seq_id'])
            return query_df

    finally:
        # Cleanup
        for f in [query_fa, target_fa, result_file]:
            if os.path.exists(f):
                os.remove(f)


# ============== Validation Set Creation ==============
def get_validset(train_df: pd.DataFrame, valid_size: int = 20) -> None:
    """
    Create validation set from megascale data using clustering.

    Args:
        train_df: Training DataFrame (should contain megascale data)
        valid_size: Number of unique sequences for validation set
    """
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
        outFa = f.name
    with tempfile.NamedTemporaryFile(suffix='', delete=False) as f:
        outResult = f.name

    # Create seq_id for clustering
    train_df = train_df.copy()
    train_df['seq_id'] = train_df.apply(
        lambda x: f"{x['dataset_name']}_{x['pdb']}_{x['chain']}",
        axis=1
    )

    # Write sequences for clustering
    seq_df = train_df[['seq_id', 'wt_seq']].drop_duplicates()
    with open(outFa, 'w') as f:
        for _, row in seq_df.iterrows():
            f.write(f">{row['seq_id']}\n{row['wt_seq']}\n")

    # Run clustering
    print(f"Running mmseqs clustering...")
    os.system(f"mmseqs easy-cluster {outFa} {outResult} tmp --min-seq-id 0.3 -c 0")

    # Parse cluster results
    cluster_df = pd.read_csv(f"{outResult}_cluster.tsv", sep="\t", header=None)
    cluster_df.columns = ['cluster_id', 'seq_id']

    # Find singleton clusters that contain only megascale sequences
    valid_seq_ids = []
    for cluster_id, group in cluster_df.groupby('cluster_id'):
        if len(group) == 1:
            seq_id = group['seq_id'].iloc[0]
            if seq_id.startswith('ms_'):
                if len(valid_seq_ids) < valid_size:
                    valid_seq_ids.append(seq_id)

    print(f"Found {len(valid_seq_ids)} validation sequences")

    # Split into train and valid
    valid_df = train_df[train_df['seq_id'].isin(valid_seq_ids)].copy()
    train_only_df = train_df[(~train_df['seq_id'].isin(valid_seq_ids)) & (train_df['dataset_name'] == "ms")].copy()

    # Save datasets
    len_seq = len(set(train_only_df['wt_seq']))
    train_only_df = train_only_df.drop(columns=['seq_id', 'dataset_name', 'dataset_type'])
    train_only_df.to_csv(f"data/train/Ms{len_seq}.csv", index=False)

    len_seq = len(set(valid_df['wt_seq']))
    valid_df = valid_df.drop(columns=['seq_id', 'dataset_name', 'dataset_type'])
    valid_df.to_csv(f"data/valid/Ms{len_seq}.csv", index=False)

    # Cleanup temp files
    for f in [outFa, outResult, f"{outResult}_cluster.tsv", f"{outResult}_rep_seq.fasta"]:
        if os.path.exists(f):
            os.remove(f)

    print(f"Train samples: {len(train_only_df)}, Valid samples: {len(valid_df)}")


# ============== Main ==============
if __name__ == "__main__":
    # Step 1: Download PDBs and extract sequences
    curated_df = get_seq(pdb_dir="data/pdb/")
    ms_df = get_megascale()
    data_df = pd.concat([curated_df, ms_df], axis=0)
    data_df = data_df.loc[(data_df['wt_seq'] != "None") & (data_df['wtAA'] != data_df['mutAA']), :]

    print(data_df.columns)
    print(data_df['dataset_name'].value_counts())
    print(data_df.groupby('dataset_name')['label'].agg(['min', 'max']))

    # Step 2: Filter training by test set and save
    os.makedirs("data/train/", exist_ok=True)
    os.makedirs("data/valid/", exist_ok=True)
    os.makedirs("data/test/", exist_ok=True)

    group_df_l = []
    test_df = data_df.loc[data_df['dataset_type'] == "test"]
    for group_name, group_df in data_df.loc[data_df['dataset_type'] == "train"].groupby("dataset_name"):
        out_df = filter_by_identity(
            query_df=group_df,
            target_df=test_df,
            cutoff=30
        )
        group_df_l.append(out_df)

    filter_df = pd.concat(group_df_l, axis=0)

    # Validate sequence consistency
    filter_df['check'] = filter_df.apply(
        lambda x: x['wt_seq'][int(x['position']) - 1] == x['wtAA'], axis=1
    )
    if not filter_df['check'].all():
        raise ValueError("filter_df['check'] exist False, check seq and position")

    test_df['check'] = test_df.apply(
        lambda x: x['wt_seq'][int(x['position']) - 1] == x['wtAA'], axis=1
    )
    if not test_df['check'].all():
        raise ValueError("test_df['check'] exist False, check seq and position")
    filter_df = filter_df.drop(columns=['check'])
    test_df = test_df.drop(columns=['check'])   

    # Save datasets
    for dataset_type, dataset_list in DATASETS.items():
        for dataset_name in dataset_list:
            if dataset_type == "train":
                filter_sub_df = filter_df.loc[
                    (filter_df['dataset_name'] == dataset_name) &
                    (filter_df['dataset_type'] == dataset_type)
                ]
                len_seq = len(filter_sub_df)
                filter_sub_df = filter_sub_df.drop(columns=['dataset_name', 'dataset_type'])
                if dataset_name == "S11278":
                    filter_sub_df.to_csv(f"data/train/C{len_seq}.csv", index=False)
                else:
                    filter_sub_df.to_csv(f"data/train/S{len_seq}.csv", index=False)
            else:
                filter_sub_df = test_df.loc[
                    (test_df['dataset_name'] == dataset_name) &
                    (test_df['dataset_type'] == dataset_type)
                ]
                filter_sub_df = filter_sub_df.drop(columns=['dataset_name', 'dataset_type'])
                filter_sub_df.to_csv(f"data/test/{dataset_name}.csv", index=False)

    # Step 3: Create validation set from megascale
    sub_df = filter_df[filter_df['dataset_type'] == "train"]
    get_validset(sub_df)
