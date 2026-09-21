import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_tiled_hard100.py"
SPEC = importlib.util.spec_from_file_location("run_tiled_hard100", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TilingTests(unittest.TestCase):
    def test_two_by_two_covers_image_edges(self):
        tiles = MODULE.tile_bounds(640, 480, 0.2, 2)
        self.assertEqual(len(tiles), 4)
        self.assertEqual(tiles[0][:2], (0, 0))
        self.assertEqual(tiles[-1][2:], (640, 480))

    def test_three_by_three_has_nine_smaller_tiles(self):
        two = MODULE.tile_bounds(640, 480, 0.2, 2)
        three = MODULE.tile_bounds(640, 480, 0.2, 3)
        self.assertEqual(len(three), 9)
        self.assertLess(three[0][2] - three[0][0], two[0][2] - two[0][0])
        self.assertEqual(three[-1][2:], (640, 480))


if __name__ == "__main__":
    unittest.main()
