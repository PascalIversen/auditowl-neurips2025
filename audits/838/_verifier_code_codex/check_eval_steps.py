"""Checks how many test batches the evaluator actually runs vs paper's '20 test iterations / ~10000 problems'.

Supports the 'eval-step-off-by-one' finding. Replicates the exact loop counting
logic from evaluation/evaluator.py:_eval (lines 31-47) without importing torch.
"""

def count_steps(max_number_of_test_steps, n_loader_batches=10**9):
    """Mirror evaluator._eval control flow for step counting only."""
    total_number_of_test_steps = 0
    executed = 0
    for step in range(n_loader_batches):
        # body runs (model.predict + metric.update)
        executed += 1
        total_number_of_test_steps += 1
        if max_number_of_test_steps and total_number_of_test_steps >= max_number_of_test_steps + 1:
            break
    return executed


if __name__ == "__main__":
    import os
    os.makedirs("out", exist_ok=True)
    cfg = 20            # main.py: DCAReasonerEvaluatorSettings(max_number_of_test_steps=20)
    batch_size = 512    # main.py
    executed = count_steps(cfg)
    problems = executed * batch_size
    with open("out/eval_steps.txt", "w") as f:
        f.write(f"configured max_number_of_test_steps = {cfg}\n")
        f.write(f"batch_size = {batch_size}\n")
        f.write(f"actual batches executed (incl. metric.update) = {executed}\n")
        f.write(f"actual test problems evaluated = {problems}\n")
        f.write("paper claims: 20 test iterations, ~10000 test problems\n")
        f.write(f"20*512 = {20*512}; 21*512 = {21*512}\n")
    print(open("out/eval_steps.txt").read())
