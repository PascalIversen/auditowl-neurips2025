"""Checks completeness of the SparseDiff repo: which training scripts, data, checkpoints,
configs, and adaptation logic exist vs. what the paper/README promise. Supports the
`missing-vqvae-grand-training`, `only-sh-system`, `hardcoded-paths`, and
`dynamic-adaptation-not-implemented` findings."""
import os, re, json, glob

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "tsinghua-fib-lab__SparseDiff")
REPO = os.path.abspath(REPO)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

def read(p):
    with open(os.path.join(REPO, p), "r", errors="ignore") as f:
        return f.read()

results = {}

# 1. Python/notebook files present
py_files = [os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True)]
nb_files = [os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "**", "*.ipynb"), recursive=True)]
results["py_files"] = sorted(py_files)
results["ipynb_files"] = sorted(nb_files)
results["config_files"] = sorted(os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "config", "*")))

# 2. Training entrypoints: what does train_sh.py train?
train = read("train_sh.py")
results["train_sh_trains_DDPM"] = "DDPM(" in train and "UNet_new" in train
results["train_sh_imports"] = re.findall(r"from model import (.+)", train)
# any script that constructs/optimizes the VQVAE or GraphModel/predictor?
all_code = ""
for p in py_files:
    all_code += read(p) + "\n"
for p in nb_files:
    nb = json.load(open(os.path.join(REPO, p)))
    for c in nb.get("cells", []):
        all_code += "".join(c.get("source", [])) + "\n"

results["vqvae_constructed_anywhere"] = bool(re.search(r"VQVAE\(", all_code))
results["graphmodel_constructed_anywhere"] = bool(re.search(r"GraphModel\(", all_code))
# Is there a backward() that involves vqvae or predictor (i.e. training them)?
results["vqvae_backward_or_optim"] = bool(re.search(r"vqvae.*\.backward|optim.*vqvae|VQVAE.*loss", all_code, re.I))
results["predictor_backward_or_optim"] = bool(re.search(r"predictor.*\.backward|optim.*predictor|GraphModel.*loss", all_code, re.I))
# how many .backward() calls total and where
results["backward_call_files"] = sorted({p for p in py_files+nb_files if ".backward(" in read(p) if p.endswith(".py")})
# notebook .backward
nb_backward = []
for p in nb_files:
    nb = json.load(open(os.path.join(REPO, p)))
    src = "".join("".join(c.get("source", [])) for c in nb.get("cells", []))
    if ".backward(" in src:
        nb_backward.append(p)
results["notebook_backward"] = nb_backward

# 3. Dynamic adaptation (latent consistency score chi_t / threshold tau) implemented?
results["mentions_cosine_similarity"] = bool(re.search(r"cosine|cos_sim|F\.cosine", all_code))
results["mentions_threshold_tau_adaptation"] = bool(re.search(r"chi_t|consistency_score|latent_consistency", all_code))
# notebook re-encoding cadence
nb = json.load(open(os.path.join(REPO, "sample_sh.ipynb")))
nb_src = "".join("".join(c.get("source", [])) for c in nb.get("cells", []))
results["notebook_uses_fixed_change_step"] = "change_step" in nb_src
results["notebook_pde_guided_call_active"] = bool(re.search(r"^\s*ddpm_samples = ema_model\.ddim_guided_sample_full_sh", nb_src, re.M))
results["notebook_pde_guided_call_commented"] = bool(re.search(r"#\s*ddpm_samples = ema_model\.ddim_guided_sample_full_sh", nb_src))
results["notebook_active_sampler"] = re.findall(r"ddpm_samples = ema_model\.(ddim_\w+)\(", nb_src)

# 4. hardcoded absolute paths
ds = read("datasets.py")
results["hardcoded_abs_paths_datasets"] = re.findall(r"'(/data5/[^']+)'", ds)

# 5. data / checkpoint dirs present?
results["data_dir_exists"] = os.path.isdir(os.path.join(REPO, "data"))
results["log_dir_exists"] = os.path.isdir(os.path.join(REPO, "log"))
results["any_npy"] = glob.glob(os.path.join(REPO, "**", "*.npy"), recursive=True)
results["any_pth"] = glob.glob(os.path.join(REPO, "**", "*.pth"), recursive=True)

# 6. Which systems have configs / dataset code?
results["dataset_names_in_get_dataset"] = re.findall(r"name == '(\w+)'", ds)

with open(os.path.join(OUT, "artifacts.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print(json.dumps(results, indent=2, default=str))
