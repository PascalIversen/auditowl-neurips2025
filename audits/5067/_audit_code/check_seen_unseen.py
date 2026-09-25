"""Check that countdown-rl test_unseen targets are disjoint from train targets, and test_seen targets overlap train (paper's seen/unseen definition). Read-only, fetches public HF parquet."""
import pandas as pd
base="https://huggingface.co/api/datasets/symoon11/countdown-rl/parquet/default"
tr=pd.read_parquet(f"{base}/train/0.parquet", columns=["extra_info"])
ts=pd.read_parquet(f"{base}/test_seen/0.parquet", columns=["extra_info"])
tu=pd.read_parquet(f"{base}/test_unseen/0.parquet", columns=["extra_info"])
def targets(df): return set(int(e["target"]) for e in df["extra_info"])
T=targets(tr); S=targets(ts); U=targets(tu)
print("train targets:", len(T), "seen targets:", len(S), "unseen targets:", len(U))
print("seen ∩ train  (should be ~all seen):", len(S & T), "/", len(S))
print("unseen ∩ train (should be 0):", len(U & T), "/", len(U))
