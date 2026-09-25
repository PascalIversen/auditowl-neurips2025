"""Documents three checks supporting findings:
(1) the implemented S-objective has an extra +1/S term vs paper Eq.14/72;
(2) the Fisher matrix is built from min-max normalized gradients (undocumented);
(3) the reported test accuracy = max over epochs on the SAME 20% holdout used as test set.
Read-only static inspection; no repo code is executed/modified."""
import re, os

ROOT = os.path.join(os.path.dirname(__file__), "..", "code", "zqy0126__OTQMS")
misc = open(os.path.join(ROOT, "misc.py")).read()
mainpy = open(os.path.join(ROOT, "main.py")).read()

out = []

# (1) extra +1/S term
m = re.search(r"equation_result = \(1/.*", misc)
out.append("OBJECTIVE LINE (misc.py):")
out.append("  " + m.group(0).strip())
out.append("  paper Eq.14/72: (d/2)*(1/(N0+s) + s^2/(N0+s)^2 * t); NO +1/S term.")
out.append("  -> code adds '+ 1/S' (==+1/S) absent from paper. PRESENT: %s" % ("+ 1/S" in m.group(0)))

# (2) min-max normalization of fisher gradients
out.append("")
out.append("FISHER NORMALIZATION (misc.py):")
norm = "normalized_fisher_value = 2 * (fisher_value - x_min) / (x_max - x_min) - 1"
out.append("  present: %s" % (norm in misc))
out.append("  paper/Algorithm-1 line 10: J = empirical mean of gradient outer products (no normalization).")

# (3) eval==test, best epoch by test metric
out.append("")
out.append("BEST-EPOCH-ON-TEST:")
out.append("  main.py computes acc on eval_loader each epoch: %s" %
           ("acc, current_epoch_eval_samples = misc.accuracy(algorithm, eval_loader, device)" in mainpy))
out.append("  check_epoch stops 'patience' epochs after the MAX acc index:")
ce = re.search(r"def check_epoch.*?return False", misc, re.S).group(0)
for ln in ce.splitlines():
    if "max_acc" in ln or "patience" in ln:
        out.append("    " + ln.strip())
out.append("  reported number = highest eval acc (paper: 'report the highest accuracies within 5 epoch early stops').")
out.append("  eval set = the 20% per-class holdout in get_trainset_and_evalset_ratioway (also called the 'test set').")

print("\n".join(out))
with open(os.path.join(os.path.dirname(__file__), "out", "objective_and_split.txt"), "w") as f:
    f.write("\n".join(out))
