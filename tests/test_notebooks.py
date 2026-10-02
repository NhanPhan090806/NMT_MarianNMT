import ast
from pathlib import Path
import subprocess
import sys

import nbformat


def test_two_kaggle_notebooks_are_valid_and_have_repo_variables():
    paths = sorted(Path(__file__).resolve().parents[1].glob("*_kaggle.ipynb"))
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
        assert 'REPO_URL = ' in combined
        assert '"git", "clone"' in combined
        assert 'resume="auto"' in combined


def test_setup_exposes_sources_in_an_already_running_interpreter(tmp_path):
    """Reproduce the logged failure without installing anything into the real venv."""
    root = Path(__file__).resolve().parents[1]
    for name in ("01_attention_mamba_kaggle.ipynb", "02_t5_transfer_learning_kaggle.ipynb"):
        notebook = nbformat.read(root / name, as_version=4)
        tree = ast.parse(notebook.cells[3].source)
        # Exercise pip and ALL helper imports, stopping before GPU selection.
        prefix = []
        for node in tree.body:
            if isinstance(node, ast.If) and "torch.cuda.is_available" in ast.unparse(node.test):
                break
            prefix.append(node)
        setup_source = ast.unparse(ast.Module(body=prefix, type_ignores=[]))
        script = '''
import importlib.util
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(sys.argv[1])
mock_site = Path(sys.argv[2])
mock_site.mkdir(exist_ok=True)
calls = []
def fake_install(command, **kwargs):
    calls.append(command)
    # Simulate pip publishing an editable-install .pth AFTER interpreter startup.
    (mock_site / 'qa_assignment.pth').write_text(str(REPO_ROOT / 'src'), encoding='utf-8')
subprocess.run = fake_install
assert importlib.util.find_spec('qa_assignment') is None
exec(sys.argv[3])
assert len(calls) == 2
assert Path(qa_assignment.__file__).resolve() == REPO_ROOT / 'src' / 'qa_assignment' / '__init__.py'
print('Live-kernel setup passed')
'''
        result = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", script,
                                 str(root), str(tmp_path), setup_source],
                                capture_output=True, text=True, encoding="utf-8", timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Live-kernel setup passed" in result.stdout
