from scripts.run_v2_ordinal_tiled import remap, tile_bounds


def test_tiling_has_four_grid_tiles_and_three_strips():
    tiles = tile_bounds(1920, 1080, 0.15)
    assert len(tiles) == 7
    assert sum(name.startswith("grid_") for name, _ in tiles) == 4
    assert sum(name.startswith("strip_") for name, _ in tiles) == 3


def test_remap_stays_normalized():
    box = remap([0.0, 0.0, 1.0, 1.0], (960, 0, 1920, 1080), 1920, 1080)
    assert box == [0.5, 0.0, 1.0, 1.0]
