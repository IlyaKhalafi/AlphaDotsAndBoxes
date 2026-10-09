import subprocess
import sys
from pathlib import Path


def test_deployed_server_and_search_do_not_import_training_frameworks():
    output = Path(__file__).resolve().parents[1] / "models/larger.npz"
    program = """
import importlib.abc
import sys
from pathlib import Path
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "ray", "gymnasium"}:
            raise AssertionError("Deployment imported " + fullname)
sys.meta_path.insert(0, BlockTraining())
from fastapi.testclient import TestClient
from alphaboxes.web.app import create_app
client = TestClient(create_app(Path(sys.argv[1])))
assert client.get("/api/health").json()["inference"] == "numpy"
game = client.post("/api/games", json={"rows": 4, "cols": 4, "demo": True}).json()
result = client.post(f"/api/games/{game['id']}/agent",
                     json={"revision": 0, "simulations": 8})
assert result.status_code == 200, result.text
assert result.json()["analysis"]["method"] == "graph_search"
assert not {"torch", "ray", "gymnasium"}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", program, str(output)], check=True, timeout=30)
