"""Verifies the ToM generalisation train/test split (TheoryOfMindTask.test_train_split):
(a) that 'names' contains substring-overlapping pairs which the substring-based name split
    (line 329 `any(name in x ...)`) handles inconsistently vs the word-boundary object split;
(b) reproduces the split disjointness on a sample of the real dataset.
Supports finding `tom-names-substring-split`."""
import csv, os, re, random, yaml

REPO = os.path.join(os.path.dirname(__file__), "..", "code", "GeodesicResearch__mars-steg")
NOUNS = os.path.join(REPO, "mars_steg", "dataset", "theory_of_mind_nouns.yaml")
CSV = os.path.join(REPO, "mars_steg", "dataset", "theory_of_mind.csv")
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

d = yaml.safe_load(open(NOUNS))
names = d["names"]

# (a) substring-containment pairs among names (split uses `name in x` substring, not \b...\b)
sub_pairs = [(a, b) for a in names for b in names if a != b and a in b]

# Simulate: if 'a' is a TRAIN name and 'b' a TEST name, a story containing only 'b'
# is assigned to test, but `train_dataset.str.contains(b)`... the assertion at line 344
# checks test-name absence in train; it does NOT prevent a TRAIN name being a substring of
# a TEST name appearing in test stories, nor guarantee word-level disjointness.
with open(os.path.join(OUT, "tom_name_substring_pairs.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["train_candidate_name", "test_candidate_name", "relation"])
    for a, b in sub_pairs:
        w.writerow([a, b, f"'{a}' is a substring of '{b}'"])

print(f"#names = {len(names)}")
print(f"name-substring-of-name pairs = {len(sub_pairs)}: {sub_pairs}")

# (b) reproduce the split's disjointness check on the real CSV (names mode), sampled.
import pandas as pd
df = pd.read_csv(CSV)
print(f"ToM dataset rows = {len(df)}; columns include: {list(df.columns)[:6]}")
random.seed(0)
shuffled = names[:]
random.shuffle(shuffled)
# pick ~half the names as 'test'
test_names = shuffled[:50]
train_names = [n for n in names if n not in test_names]
# substring assignment exactly as the repo does (line 329)
sample = df.head(3000)
test_mask = sample["infilled_story"].apply(lambda x: any(n in x for n in test_names))
train_rows = sample[~test_mask]
# repo assertion (line 344): no test name substring in any train story
violations = [n for n in test_names
              if train_rows["infilled_story"].str.contains(re.escape(n), regex=True).any()]
print(f"sample rows = {len(sample)}; train rows = {len(train_rows)}; test rows = {test_mask.sum()}")
print(f"test-name substrings still present in train stories (repo assertion would FAIL on these): {violations[:10]}")
with open(os.path.join(OUT, "tom_split_check.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_rows", "train_rows", "test_rows", "n_test_name_violations_in_train"])
    w.writerow([len(sample), len(train_rows), int(test_mask.sum()), len(violations)])
