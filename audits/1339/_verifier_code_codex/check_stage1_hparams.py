"""Compares the stage-1 (Driving-Thinking) hyperparameters the paper reports in Sec. 4.1 against the only released stage-1 config/optimiser (supports finding: stage1-hparams-differ-from-paper)."""
import os, re, yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code",
                                    "AIR-DISCOVER__E-cubed-AD__E-VAD"))
CFG = os.path.join(REPO, "projects/eeg_vedio/cfgs/train_config.yaml")
TRAIN = os.path.join(REPO, "projects/eeg_vedio/src/run_model/train.py")
MODEL = os.path.join(REPO, "projects/eeg_vedio/src/models/contrastive_model.py")

cfg = yaml.safe_load(open(CFG))
m = cfg["model"]
train_src = open(TRAIN).read().splitlines()
model_src = open(MODEL).read().splitlines()

def find(lines, needle, path):
    for i, l in enumerate(lines, 1):
        if needle in l:
            return f"{path}:{i}", l.strip()
    return "(not found)", ""

opt_loc, opt_line = find(train_src, "optim.Adam(", "projects/eeg_vedio/src/run_model/train.py")
tmp_loc, tmp_line = find(model_src, "self.temperature", "projects/eeg_vedio/src/models/contrastive_model.py")
sim_loc, sim_line = find(model_src, "/ self.temperature", "projects/eeg_vedio/src/models/contrastive_model.py")

# dropout: VideoEncoder does `config["dropout"] if "dropout" in config else None`
video_cfg = m["video_encoder"]
dropout_in_cfg = "dropout" in video_cfg

rows = [
    ("epochs",            "120",                  str(m["epochs"]),                 "cfgs/train_config.yaml:12"),
    ("batch size",        "16",                   str(m["batch_size"]),             "cfgs/train_config.yaml:13"),
    ("learning rate",     "2e-5",                 str(m["learning_rate"]),          "cfgs/train_config.yaml:14"),
    ("weight decay",      "1e-5",                 "0 (torch.optim.Adam default; no weight_decay arg passed)", opt_loc),
    ("dropout",           "0.01",                 "absent from config -> VideoEncoder sets self.dropout=None -> create_mlp adds no Dropout"
                                                  if not dropout_in_cfg else str(video_cfg["dropout"]),
                                                  "cfgs/train_config.yaml (video_encoder block, lines 24-28)"),
    ("clip duration",     "2 s",                  f'{m.get("data_duration", cfg["data"]["data_duration"])} s',
                                                  "cfgs/train_config.yaml:6"),
    ("clip frame rate",   "2 fps",                f'10 fps (version_identifier={cfg["version_identifier"]!r})',
                                                  "cfgs/train_config.yaml:9"),
    ("temperature",       "learnable log-temperature beta", f'fixed constant {m["temperature"]} from config',
                                                  f"cfgs/train_config.yaml:15 / {sim_loc}"),
    ("trainable params",  "both backbones + adapters (lr ratio 1:1)",
                          f'fine_tuning_policy={m["fine_tuning_policy"]!r} -> only the un-restored keys train',
                          "cfgs/train_config.yaml:16 + contrastive_model.py:33-44"),
    ("adapter width",     "two-layer MLP -> 200-d", f'mlp_layers={video_cfg["mlp_layers"]}', "cfgs/train_config.yaml:28"),
]

print("Stage-1 (Driving-Thinking) hyperparameters: paper Sec. 4.1 vs released code\n")
w = (18, 42, 62)
print(f"{'SETTING':{w[0]}} {'PAPER (Sec. 4.1 / Sec. 3.2)':{w[1]}} {'CODE':{w[2]}} EVIDENCE")
print("-" * 170)
n_mismatch = 0
for name, paper, code, ev in rows:
    print(f"{name:{w[0]}} {paper:{w[1]}} {code:{w[2]}} {ev}")
print()
print("Verbatim optimiser line   ->", opt_loc, ":", opt_line)
print("Verbatim temperature line ->", sim_loc, ":", sim_line)
print("\nvideo_encoder config block keys:", list(video_cfg.keys()),
      "  ('dropout' present? ->", dropout_in_cfg, ")")

# does train.py build any train/val/test split?
split_hits = [f"{i}: {l.strip()}" for i, l in enumerate(train_src, 1)
              if re.search(r"split|val_|valid|test_loader|random_split|Subset", l)]
print("\ntrain.py lines mentioning any split/validation construct:",
      split_hits if split_hits else "NONE")
print("DataLoader construction ->", *find(train_src, "DataLoader(", "projects/eeg_vedio/src/run_model/train.py"))
print("  -> the full RealCarDataset is used as the training loader; the paper's")
print("     80:10:10 train/val/test partition is not constructed anywhere in the released stage-1 code.")
