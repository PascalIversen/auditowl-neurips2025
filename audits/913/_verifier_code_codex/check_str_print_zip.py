"""Confirms str_print zips 3 split names against a length-2 metric tensor, mislabeling outputs.
Supports finding `str-print-mislabel`."""
labels = ['train', 'val', 'test']
metric_mean = [0.10, 0.20]  # (val, test) per get_fold_metrics
metric_std  = [0.01, 0.02]
pairs = list(zip(labels, metric_mean, metric_std))
print("zip pairs:", pairs)
print("=> printed 'train' shows the val value; 'val' shows the test value; 'test' dropped")
