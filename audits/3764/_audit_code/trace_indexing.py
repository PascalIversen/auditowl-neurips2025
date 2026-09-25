"""Traces the loop-variable shadowing and list indexing in DNAEdit_SD3/FLUX
to check whether the inner `for i in range(k)` corrupts the outer enumerate index
and whether dx_lst/v_lst indexing in the editing loop is consistent.
Supports finding: dna-loop-var-shadowing (no GPU needed; pure control-flow trace)."""
import torch

# emulate: T_steps timesteps -> retrieve_timesteps gives T_steps values
# then timesteps = cat([timesteps, 0]); inver = flip(timesteps)
T_steps = 28
jmp = 4  # T_start (FLUX)
# fabricate a descending timestep schedule (1000..~36) like FLUX after retrieve_timesteps
ts = torch.linspace(1000, 1000/T_steps, T_steps)
timesteps = torch.cat([ts, torch.tensor([0.0])])  # len T_steps+1
inver = torch.flip(timesteps, dims=[0])            # ascending then... flip of [desc..,0] => [0, asc...]

print("len(timesteps) =", len(timesteps))
print("len(inver) =", len(inver))

# --- inversion loop ---
appended = 0
outer_i_seen = []
for i,(t_curr,t_prev) in enumerate(zip(inver[:-1], inver[1:])):
    if len(inver)-1-i == jmp:
        print(f"BREAK at outer enumerate i={i} (len-1-i={len(inver)-1-i}==jmp)")
        break
    outer_i_seen.append(i)
    # inner loop shadows i
    k = 1
    for i in range(k):
        pass  # after this, name 'i' == k-1 == 0
    appended += 1  # one append per outer iteration
print("number of appends to dx_lst/v_lst/last_lst =", appended)
print("outer indices that ran:", outer_i_seen)

# After loops, lists reversed: dx_lst[::-1]; length = appended
# editing loop:
print("\n--- editing loop indexing ---")
max_idx_used = -1
editing_iters = 0
for i,(t_curr,t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
    if i < jmp:
        continue
    idx = i - jmp
    editing_iters += 1
    max_idx_used = max(max_idx_used, idx)
print("editing iterations =", editing_iters)
print("max dx_lst index used =", max_idx_used, " ; list length =", appended)
print("INDEX OK?" , max_idx_used < appended)
