"""Read-only: compare saved spin results (results_exp_100000.json) against paper Table 6 (M=100,000).
Supports findings: spin-train-vs-test-loss, units-bits-vs-nats, lambda-table-mismatch.
Does NOT modify the repo; only reads its committed result JSON.
"""
import json, os
import numpy as np

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "EkMeasurable__Learnability_Ipred")
P = os.path.join(REPO, "results_data", "spin_xps", "results_exp_100000.json")
r = json.load(open(P))
ulc = r["universal_learning_curve"]          # np.diff(predictive_info_1), in BITS (log2 entropy)
evo = r["evoRate_empirical"]                  # also bits

# Paper Table 6 (M=100,000): (k, R_LSTM, R_MLP, Lambda_hat, EvoRate)
paper = {
    1:(0.6897,0.6933,0.3213,0.0056),
    2:(0.5789,0.6070,0.2061,0.1261),
    3:(0.6036,0.5745,0.1506,0.1832),
    5:(0.5146,0.5383,0.0964,0.2379),
    10:(0.4901,0.5186,0.0479,0.2861),
}

out = []
out.append("k, paper_R_LSTM, json_LSTM_train(last100), json_LSTM_val[-1], json_LSTM_minVal")
for k in [1,2,3,5,10]:
    d = r["models"]["LSTM"][f"k_{k}"]
    tl = np.mean(d["train_losses"][-100:])
    vlast = d["val_losses"][-1]
    vmin = min(d["val_losses"])
    out.append(f"{k}, {paper[k][0]:.4f}, {tl:.4f}, {vlast:.4f}, {vmin:.4f}")

out.append("")
out.append("k, paper_Lambda_hat, json_ulc(bits), json_ulc*ln2(nats??), paper_EvoRate, json_evoRate(bits)")
for k in [1,2,3,5,10]:
    u = ulc[k-1]
    out.append(f"{k}, {paper[k][2]:.4f}, {u:.4f}, {u*np.log(2):.4f}, {paper[k][3]:.4f}, {evo[k-1]:.4f}")

txt = "\n".join(out)
print(txt)
open(os.path.join(os.path.dirname(__file__),"out","spin_table_compare.txt"),"w").write(txt+"\n")
