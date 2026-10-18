"""
Smoke test to execute all code cells in Advanced_Fake_News_Detection.ipynb sequentially
in a clean global environment, catching any undefined variables or runtime exceptions.
"""

import sys
import os
import nbformat
import matplotlib
matplotlib.use("Agg")

def run_notebook_cells(nb_path="Advanced_Fake_News_Detection.ipynb"):
    print(f"Loading notebook: {nb_path}...")
    nb = nbformat.read(nb_path, as_version=4)

    env = {"__name__": "__main__"}

    # Mock display if running outside IPython
    def mock_display(*args):
        for a in args:
            print(a)
    env["display"] = mock_display

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    print(f"Found {len(code_cells)} code cells to execute.\n")

    for idx, cell in enumerate(code_cells, start=1):
        source = cell.source
        header = f"--- Executing Cell {idx}/{len(code_cells)} ---"
        print(header)
        try:
            exec(source, env)
            print(f"[OK] Cell {idx} completed successfully.\n")
        except Exception as e:
            print(f"\n[ERROR in Cell {idx}]: {e}\n")
            print("Cell Source:")
            print(source[:500])
            raise

    print("ALL CODE CELLS EXECUTED SUCCESSFULLY WITHOUT ERRORS!")

if __name__ == "__main__":
    run_notebook_cells()
