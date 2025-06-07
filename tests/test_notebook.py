import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path


def test_notebook_execution(tmp_path):
    nb_path = Path(__file__).resolve().parents[1] / "My first example.ipynb"
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {'metadata': {'path': nb_path.parent}})
