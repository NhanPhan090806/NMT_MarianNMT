import ast
from pathlib import Path

import nbformat


def test_two_kaggle_notebooks_are_valid_and_have_empty_repo_variables():
    paths = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
    assert len(paths) == 2
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert cell.outputs == []
                assert cell.execution_count is None
        combined = "\n".join(cell.source for cell in notebook.cells)
        assert 'REPO_URL = ""' in combined
        assert '"git", "clone"' in combined
        assert 'resume="auto"' in combined

