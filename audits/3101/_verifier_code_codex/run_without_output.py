"""Run the copied check script while diverting its incidental JSON output."""
import builtins
import io
import os
import runpy

SCRIPT = os.path.join(os.path.dirname(__file__), "run_kl_table.py")
OUTPUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "out", "kl_table.json"))
real_open = builtins.open


def diverted_open(file, mode="r", *args, **kwargs):
    if os.path.normpath(os.fspath(file)) == OUTPUT and "w" in mode:
        return io.StringIO()
    return real_open(file, mode, *args, **kwargs)


builtins.open = diverted_open
try:
    runpy.run_path(SCRIPT, run_name="__main__")
finally:
    builtins.open = real_open
