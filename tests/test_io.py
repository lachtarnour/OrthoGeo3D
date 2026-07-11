import tempfile
import unittest
from pathlib import Path

from src.utils.io import load_mesh


class TestIO(unittest.TestCase):
    def test_load_mesh_triangulates_quad_once(self) -> None:
        content = "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 1 1 0",
                "v 0 1 0",
                "f 1 2 3 4",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "quad.obj"
            path.write_text(content, encoding="utf-8")

            mesh = load_mesh(path)

        self.assertEqual(mesh["vertices"].shape, (4, 3))
        self.assertEqual(mesh["faces"].shape, (2, 3))
        self.assertEqual(mesh["faces"].tolist(), [[0, 1, 2], [0, 2, 3]])


if __name__ == "__main__":
    unittest.main()
