"""Data augmentation by TR (Time Reversal) or TP (Transitive Property)."""

import os
from typing import Optional

import pandas as pd
from tqdm import tqdm


def tr_augment(data_df: pd.DataFrame,
               out_prefix: str,
               save_each: bool = False,
               sample_n: Optional[int] = None) -> None:
    """Data augmentation by time reversal (TR)."""
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    if sample_n:
        data_df['seq_id'] = data_df['pdb'] + "_" + data_df['chain']
        sample_n //= 2
        group_counts = data_df['seq_id'].value_counts()
        sample_counts = (group_counts / group_counts.sum() * sample_n).round().astype(int)
        sample_counts.iloc[0] += sample_n - sample_counts.sum()
        dfs = [data_df[data_df['seq_id'] == g].sample(n=min(n, len(data_df[data_df['seq_id'] == g])), random_state=1998)
               for g, n in sample_counts.items()]
        data_df = pd.concat(dfs, ignore_index=True)
        data_df = data_df.drop(columns=["seq_id"])

    data_d_df = data_df.copy()
    data_r_df = data_df.copy()
    data_r_df['wtAA'] = data_d_df['mutAA']
    data_r_df['mutAA'] = data_d_df['wtAA']
    data_r_df['wt_seq'] = data_d_df['mut_seq']
    data_r_df['mut_seq'] = data_d_df['wt_seq']
    data_r_df["label"] = -data_d_df["label"]

    if save_each:
        data_d_df.to_csv(f"{out_prefix}_d.csv", index=False)
        data_r_df.to_csv(f"{out_prefix}_r.csv", index=False)
    else:
        pd.concat([data_d_df, data_r_df], axis=0).to_csv(f"{out_prefix}.csv", index=False)


def tp_augment(data_df: pd.DataFrame,
               out_prefix: str,
               sample_n: Optional[int] = None) -> None:
    """Data augmentation by transitive property (TP)."""
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    out_d_l = []
    for _, data_group_df in tqdm(data_df.groupby(['pdb', 'chain', 'position']), total=data_df.groupby(['pdb', 'chain', 'position']).ngroups):
        group_records = data_group_df.to_dict(orient='records')
        n = len(group_records)

        for i in range(n):
            for j in range(n):
                row_i, row_j = group_records[i], group_records[j]
                if i == j:
                    out_d_l.append(row_i)
                    flipped = row_i.copy()
                    flipped.update({'wtAA': row_i['mutAA'], 'mutAA': row_i['wtAA'],
                                    'wt_seq': row_i['mut_seq'], 'mut_seq': row_i['wt_seq'], "label": -row_i["label"]})
                    out_d_l.append(flipped)
                else:
                    new_row = row_i.copy()
                    new_row.update({'wtAA': row_i['mutAA'], 'mutAA': row_j['mutAA'],
                                    'wt_seq': row_i['mut_seq'], 'mut_seq': row_j['mut_seq'], "label": row_j["label"] - row_i["label"]})
                    out_d_l.append(new_row)

    result_df = pd.DataFrame(out_d_l)

    if sample_n:
        result_df['seq_id'] = result_df['pdb'] + "_" + result_df['chain']
        group_counts = result_df['seq_id'].value_counts()
        sample_counts = (group_counts / group_counts.sum() * sample_n).round().astype(int)
        sample_counts.iloc[0] += sample_n - sample_counts.sum()
        dfs = [result_df[result_df['seq_id'] == g].sample(n=min(n, len(result_df[result_df['seq_id'] == g])), random_state=1998)
               for g, n in sample_counts.items()]
        result_df = pd.concat(dfs, ignore_index=True)
        result_df = result_df.drop(columns=["seq_id"])

    result_df.to_csv(f"{out_prefix}.csv", index=False)


if __name__ == "__main__":
    # C4238
    inputPrefix = "/home/DATA_2/hujiameng/work/5.stability_v7/data/train/C4238"
    tr_augment(pd.read_csv(f"{inputPrefix}.csv"), f"{inputPrefix}_tr")

    # S2403
    inputPrefix = "/home/DATA_2/hujiameng/work/5.stability_v7/data/train/S2403"
    tr_augment(pd.read_csv(f"{inputPrefix}.csv"), f"{inputPrefix}_tr")

    # Ms154 TP augmentation
    sample_n = 60000
    inputPrefix = "/home/DATA_2/hujiameng/work/5.stability_v7/data/train/Ms154"
    tp_augment(pd.read_csv(f"{inputPrefix}.csv"), f"{inputPrefix}_tp_{sample_n}", sample_n=sample_n)

    # Validation
    sample_n = 1000
    inputPrefix = "/home/DATA_2/hujiameng/work/5.stability_v7/data/valid/Ms20"
    tr_augment(pd.read_csv(f"{inputPrefix}.csv"), f"{inputPrefix}_{sample_n//2}_tr", save_each=True, sample_n=sample_n)

    # Test datasets
    for id_c in ['ssym', 'p53', 'myoglobin', 'S669']:
        inputPrefix = f"/home/DATA_2/hujiameng/work/5.stability_v7/data/test/{id_c}"
        tr_augment(pd.read_csv(f"{inputPrefix}.csv"), f"{inputPrefix}_tr", save_each=True)