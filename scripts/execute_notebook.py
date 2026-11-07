"""Execute in a fresh local kernel, saving partial output even on failure."""

from pathlib import Path
import json
import sys
import os
import faulthandler
import asyncio

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if "--debug" in sys.argv:
    faulthandler.dump_traceback_later(45, repeat=True)
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".jupyter-runtime")
os.environ["IPYTHONDIR"] = str(ROOT / ".ipython")
kernel = ROOT / ".venv/share/jupyter/kernels/ml-cep"
kernel.mkdir(parents=True, exist_ok=True)
(kernel / "kernel.json").write_text(
    json.dumps(
        {
            "argv": [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ],
            "display_name": "ML-CEP (.venv Python 3.12)",
            "language": "python",
        }
    )
)
path = ROOT / "notebooks/audio_emotion_analysis.ipynb"
notebook = nbformat.read(path, as_version=4)


def announce_cell(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print(
            f"Executing cell {cell_index + 1}/{len(notebook.cells)}: {cell.source.splitlines()[0]}",
            flush=True,
        )


client = NotebookClient(
    notebook,
    timeout=7200,
    kernel_name="ml-cep",
    resources={"metadata": {"path": str(ROOT)}},
    on_cell_start=announce_cell,
)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
    faulthandler.cancel_dump_traceback_later()
print("Fresh-kernel execution complete")
