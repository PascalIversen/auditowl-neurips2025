"""Checks the confidence-interval helper ci2() used by both plot scripts: the t-quantile sign and the hardcoded sample size. Supports finding: ci-formula-wrong-quantile-and-n."""
import math
import os

from scipy.stats import t

OUT = os.path.join(os.path.dirname(__file__), "out", "ci_formula.txt")
lines = []


def ci2(mean, std, n, conf):
    # verbatim re-implementation of plots/run_*_plot.py ci2()
    t_value = t.ppf(1 - conf, n - 1)
    margin_error = t_value * std / math.sqrt(n)
    return mean - margin_error, mean + margin_error, t_value


lines.append("Paper Figures 1-4 captions all state '95 %c.i.'")
lines.append("")
lines.append("ci2 in plots/run_estimation_err_plot.py uses conf=0.90, n passed = 10 or 20")
lines.append("ci2 in plots/run_regret_plot.py     uses conf=0.85, n passed = num_experiments")
lines.append("")

# Estimation-error plot call: conf=0.90, n=confidence=10
for (conf, n, tag) in [(0.90, 10, "est-err main (n hardcoded=10)"),
                       (0.90, 20, "est-err custom (n hardcoded=20)"),
                       (0.85, 5, "regret (n=num_experiments=5)"),
                       (0.85, 3, "regret pomdp1 (n=3)")]:
    lo, hi, tv = ci2(mean=1.0, std=1.0, n=n, conf=conf)
    # what the *correct* two-sided 95% t-multiplier would be for this n
    correct_t = t.ppf(0.975, n - 1)
    lines.append(f"[{tag}] conf={conf} n={n}: "
                 f"t.ppf(1-conf={1-conf:.2f}, df={n-1}) = {tv:.4f}  "
                 f"(SIGN {'NEGATIVE -> band inverted' if tv < 0 else 'positive'}); "
                 f"correct two-sided 95% t = {correct_t:.4f}")

lines.append("")
lines.append("CONCLUSIONS:")
lines.append("1) t.ppf(1-conf, ...) with conf>0.5 returns a NEGATIVE quantile, so")
lines.append("   lower_bound = mean - t*std/sqrt(n) is actually ABOVE the mean and")
lines.append("   upper_bound BELOW it: the shaded 'CI' band is inverted/swapped.")
lines.append("2) The confidence level used (0.90 / 0.85) does not match the captioned 95%.")
lines.append("3) The 'n' fed to ci2 in the estimation plot is a hardcoded constant")
lines.append("   (confidence=10 or 20), not the true number of runs in the data.")

txt = "\n".join(lines) + "\n"
with open(OUT, "w") as f:
    f.write(txt)
print(txt)
