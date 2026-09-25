#!/usr/bin/env python3
"""Documents the temperature/scale discrepancy between paper Eq.14 (L_mutual) and
the code's DistillKL(T=4.0) applied to tau=16 cosine logits.
Supports finding `lmutual-distill-temperature`. Read-only; writes out/lmutual_temp.txt.

Paper Eq.11:  L = (1-beta) L_high + beta L_low + gamma L_mutual,  gamma=0.5
Paper Eq.14:  L_mutual = KL( Phi(tau*S_low) || Phi(tau*S_high) ) + KL( Phi(tau*S_high) || Phi(tau*S_low) )
              Phi = softmax, tau = 16 already baked into the cosine logits.
Code (helper.py:29,33): mutual_loss = crit(logits1,logits2)+crit(logits2,logits1);
              loss += complex_weight * mutual_loss,  complex_weight=gamma=0.5
Code DistillKL(T):  p_s=log_softmax(y/T); p_t=softmax(y/T);
              loss = kl_div(p_s,p_t,reduce=sum) * T**2 / batch
where y = logits already scaled by tau=16, and T = 4.0.
"""
import os

tau = 16.0   # args.temperature, applied to cosine sim before logits
T = 4.0      # DistillKL temperature hard-coded in fscil_trainer.py:59
gamma = 0.5  # args.complex_weight (paper gamma)

lines = []
lines.append("Paper Eq.14 softmax sharpness on cosine sim S:  exp(tau * S) -> temperature on S = 1/tau = %.4f" % (1.0/tau))
lines.append("Code effective sharpness:  logits = tau*S, then softmax(logits / T) = exp((tau/T) * S)")
lines.append("  -> effective temperature on S = T/tau = %.4f  (i.e. exp(%.1f * S) instead of exp(%.1f * S))" % (T/tau, tau/T, tau))
lines.append("")
lines.append("So the implemented L_mutual softens the two distributions by an extra factor T=%.1f" % T)
lines.append("relative to paper Eq.14 (which uses the full tau=%.0f sharpness)." % tau)
lines.append("")
lines.append("Magnitude: DistillKL multiplies the summed-over-classes KL by T**2 = %.0f and divides by batch." % (T**2))
lines.append("Paper Eq.14 has no T**2 factor. Combined with gamma=%.1f the code's mutual term carries an" % gamma)
lines.append("implicit weight of gamma*T**2 = %.1f x the per-sample KL, vs gamma=%.1f in the paper's Eq.11." % (gamma*T**2, gamma))
lines.append("")
lines.append("CONCLUSION: code computes a distillation-KL at effective temperature 4 on the cosine similarities,")
lines.append("scaled by 16; paper Eq.14 specifies a plain bidirectional KL at the full tau=16 sharpness. Valid")
lines.append("KL either way -> category 'difference' (description vs implementation), not a bug.")

OUT = os.path.join(os.path.dirname(__file__), "out", "lmutual_temp.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
