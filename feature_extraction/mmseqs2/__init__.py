"""
MMSEQS MSA generation module
Supports GPU and CPU versions
"""

import os
import subprocess
import time
from typing import Tuple, Optional, List, Dict


# ============== Configuration ==============

DEFAULT_TARGET_DB_GPU = "/home/DATA_2/hujiameng/source/mmseqs/gpu/uniref90_DB_gpu"
DEFAULT_TARGET_DB_CPU = "/home/DATA_2/hujiameng/source/mmseqs/avx2/uni90_DB"
MMSEQS_PATH_GPU = "/home/DATA_2/hujiameng/tools/mmseqs-gpu/last/mmseqs/bin/mmseqs"
MMSEQS_PATH_CPU = "/home/DATA_2/hujiameng/tools/mmseqs/bin/mmseqs"
DEFAULT_MAX_SEQS = 1000
DEFAULT_THREADS = 24
DEFAULT_GPU_ID = 0
DEFAULT_TIMEOUT = 3600  # seconds


def get_msa_gpu(
    sequences: List[Dict[str, str]],
    outdir: str,
    targetDB: str = DEFAULT_TARGET_DB_GPU,
    max_seqs: int = DEFAULT_MAX_SEQS,
    threads: int = DEFAULT_THREADS,
    mmseqs_path: Optional[str] = MMSEQS_PATH_GPU,
    timeout: int = DEFAULT_TIMEOUT
) -> Tuple[bool, float]:
    """
    Generate MSA using GPU version of MMSEQS

    Args:
        sequences: List of sequence dicts with 'id' and 'sequence' keys
                   Example: [{"id": "seq1", "sequence": "MVLSPADKTN..."}, ...]
        outdir: Output directory
        targetDB: Target database path
        max_seqs: Maximum number of sequences
        threads: Number of threads
        mmseqs_path: Path to mmseqs-gpu binary (default: use system mmseqs-gpu)
        timeout: Timeout in seconds for each command

    Returns:
        Tuple of (success: bool, elapsed_time: float)
    """
    # Input validation
    if not sequences:
        raise ValueError("sequences cannot be empty")
    for i, seq in enumerate(sequences):
        if not seq.get("id") or not seq.get("sequence"):
            raise ValueError(f"sequences[{i}]: 'id' and 'sequence' are required")
    if max_seqs <= 0:
        raise ValueError("max_seqs must be positive")
    if threads <= 0:
        raise ValueError("threads must be positive")

    # Use custom path or find system executable
    mmseqs_cmd = mmseqs_path if mmseqs_path else "mmseqs-gpu"

    # Resolve paths
    outdir = os.path.abspath(outdir)
    out_fa = os.path.join(outdir, "query.fa")
    query_db = os.path.join(outdir, "query_DB")
    out_msa = os.path.join(outdir, "msa.fa")

    os.makedirs(outdir, exist_ok=True)

    # Write query sequences to FASTA
    if not os.path.exists(out_fa):
        with open(out_fa, "w") as f:
            for seq in sequences:
                f.write(f">{seq['id']}\n{seq['sequence']}\n")

    print(f"===={len(sequences)} sequences")
    start_time = time.time()

    try:
        # Step 1: Create query database
        print("  [1/3] Creating query database...")
        ret = subprocess.run(
            [mmseqs_cmd, "createdb", out_fa, query_db, "--threads", str(threads)],
            capture_output=False, timeout=timeout
        )
        if ret.returncode != 0:
            print(f"ERROR: failed to create database")
            return False, time.time() - start_time
        print("  [1/3] Done")

        # Step 2: Search
        print("  [2/3] Searching database...")
        ret = subprocess.run(
            [mmseqs_cmd, "search", query_db, targetDB,
             os.path.join(outdir, "result"), os.path.join(outdir, "tmp"),
             "--max-seqs", str(max_seqs), "--gpu", "1", "--threads", str(threads)],
            capture_output=False, timeout=timeout
        )
        if ret.returncode != 0:
            print(f"ERROR: failed to search")
            return False, time.time() - start_time
        print("  [2/3] Done")

        # Step 3: Convert to MSA
        print("  [3/3] Converting to MSA...")
        ret = subprocess.run(
            [mmseqs_cmd, "result2msa", query_db, targetDB,
             os.path.join(outdir, "result"), out_msa,
             "--msa-format-mode", "5", "--threads", str(threads)],
            capture_output=False, timeout=timeout
        )
        if ret.returncode != 0:
            print(f"ERROR: failed to convert to MSA")
            return False, time.time() - start_time
        print("  [3/3] Done")

    except subprocess.TimeoutExpired:
        print(f"ERROR: command timed out after {timeout}s")
        return False, time.time() - start_time
    except Exception as e:
        print(f"ERROR: {e}")
        return False, time.time() - start_time

    elapsed_time = time.time() - start_time
    print(f"MSA saved to: {out_msa} ({elapsed_time:.2f}s)")

    return True, elapsed_time


def get_msa_cpu(
    sequences: List[Dict[str, str]],
    outdir: str,
    targetDB: str = DEFAULT_TARGET_DB_CPU,
    max_seqs: int = DEFAULT_MAX_SEQS,
    threads: int = DEFAULT_THREADS,
    mmseqs_path: Optional[str] = MMSEQS_PATH_CPU,
    timeout: int = DEFAULT_TIMEOUT
) -> Tuple[bool, float]:
    """
    Generate MSA using CPU version of MMSEQS

    Args:
        sequences: List of sequence dicts with 'id' and 'sequence' keys
                   Example: [{"id": "seq1", "sequence": "MVLSPADKTN..."}, ...]
        outdir: Output directory
        targetDB: Target database path
        max_seqs: Maximum number of sequences
        threads: Number of threads
        mmseqs_path: Path to mmseqs binary (default: use system mmseqs)
        timeout: Timeout in seconds for each command

    Returns:
        Tuple of (success: bool, elapsed_time: float)
    """
    # Input validation
    if not sequences:
        raise ValueError("sequences cannot be empty")
    for i, seq in enumerate(sequences):
        if not seq.get("id") or not seq.get("sequence"):
            raise ValueError(f"sequences[{i}]: 'id' and 'sequence' are required")
    if max_seqs <= 0:
        raise ValueError("max_seqs must be positive")
    if threads <= 0:
        raise ValueError("threads must be positive")

    # Use custom path or find system executable
    mmseqs_cmd = mmseqs_path if mmseqs_path else "mmseqs"

    # Resolve paths
    outdir = os.path.abspath(outdir)
    out_fa = os.path.join(outdir, "query.fa")
    query_db = os.path.join(outdir, "query_DB")
    out_msa = os.path.join(outdir, "msa.fa")

    os.makedirs(outdir, exist_ok=True)

    # Write query sequences to FASTA
    if not os.path.exists(out_fa):
        with open(out_fa, "w") as f:
            for seq in sequences:
                f.write(f">{seq['id']}\n{seq['sequence']}\n")

    print(f"===={len(sequences)} sequences")
    start_time = time.time()

    try:
        # Step 1: Create query database
        print("  [1/3] Creating query database...")
        ret = subprocess.run(
            [mmseqs_cmd, "createdb", out_fa, query_db, "--threads", str(threads)],
            capture_output=False, timeout=timeout
        )
        if ret.returncode != 0:
            print(f"ERROR: failed to create database")
            return False, time.time() - start_time
        print("  [1/3] Done")

        # Step 2: Search
        print("  [2/3] Searching database...")
        ret = subprocess.run(
            [mmseqs_cmd, "search", query_db, targetDB,
             os.path.join(outdir, "result"), os.path.join(outdir, "tmp"),
             "--max-seqs", str(max_seqs), "--threads", str(threads)],
            capture_output=False, timeout=timeout
        )
        if ret.returncode != 0:
            print(f"ERROR: failed to search")
            return False, time.time() - start_time
        print("  [2/3] Done")

        # Step 3: Convert to MSA
        print("  [3/3] Converting to MSA...")
        ret = subprocess.run(
            [mmseqs_cmd, "result2msa", query_db, targetDB,
             os.path.join(outdir, "result"), out_msa,
             "--msa-format-mode", "5", "--threads", str(threads)],
            capture_output=False, timeout=timeout
        )
        if ret.returncode != 0:
            print(f"ERROR: failed to convert to MSA")
            return False, time.time() - start_time
        print("  [3/3] Done")

    except subprocess.TimeoutExpired:
        print(f"ERROR: command timed out after {timeout}s")
        return False, time.time() - start_time
    except Exception as e:
        print(f"ERROR: {e}")
        return False, time.time() - start_time

    elapsed_time = time.time() - start_time
    print(f"MSA saved to: {out_msa} ({elapsed_time:.2f}s)")

    return True, elapsed_time


# Default to GPU version
get_msa = get_msa_gpu


# ============== Example ==============
if __name__ == "__main__":
    # Test GPU version with multiple sequences
    sequences = [
        {"id": "seq1", "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"},
        {"id": "seq2", "sequence": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST"}
    ]
    success, elapsed = get_msa_gpu(
        sequences=sequences,
        outdir="/home/DATA_2/hujiameng/test/mmseqs2Test/test_output_multi"
    )
    print(f"GPU test: success={success}, time={elapsed:.2f}s")

    # Test CPU version with multiple sequences
    success, elapsed = get_msa_cpu(
        sequences=sequences,
        outdir="/home/DATA_2/hujiameng/test/mmseqs2Test/test_output_multi_cpu"
    )
    print(f"CPU test: success={success}, time={elapsed:.2f}s")
