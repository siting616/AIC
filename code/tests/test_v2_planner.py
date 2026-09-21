from src.v2.query_parser_v2 import parse_query
from src.v2.query_taxonomy import classify_query, operator_chain
from src.v2.scene_planner import build_scene_plan
from src.v2.ordinal_solver import build_ordinal_proposals, canonical_target, clean_candidates


def test_tshirt_does_not_trigger_infrared():
    parsed = parse_query("A young man in a black T-shirt taking a photo")
    assert parsed["thermal"]["enabled"] is False


def test_composed_depth_part_chain():
    parsed = parse_query("The right ear of the bear closest to the camera")
    assert set(classify_query(parsed)) == {"depth", "part"}
    chain = operator_chain(parsed)
    assert chain.index("depth_filter") < chain.index("part_localize")


def test_horizontal_ordinal_is_structured():
    parsed = parse_query("From left to right, the fourth stone pier")
    assert parsed["ordinal"] == {
        "enabled": True,
        "type": "spatial",
        "axis": "x",
        "direction": "left_to_right",
        "index": 4,
    }


def test_plan_never_allows_override_in_stage_one():
    records = {
        "000023_001": {"visible": "Images/visible/000023.png", "query": "the first stone pier from the left"},
        "000023_002": {"visible": "Images/visible/000023.png", "query": "the second stone pier from the left"},
    }
    plan = build_scene_plan(records)
    assert plan["scene_count"] == 1
    assert plan["route_counts"]["ordinal"] == 2
    assert all(not item["override_allowed"] for item in plan["scenes"]["000023"])


def test_reference_ordinal_does_not_route_target():
    parsed = parse_query("The stone pier behind the fence to the right of the fourth stone pier from the left")
    assert parsed["ordinal"]["enabled"] is False


def test_canonical_target_removes_ordinal_words():
    assert canonical_target("From left to right, the fourth stone pier") == "stone pier"


def test_candidate_cleanup_deduplicates_large_overlap():
    candidates = [
        {"bbox": [0.10, 0.10, 0.20, 0.30], "score": 0.9},
        {"bbox": [0.101, 0.10, 0.201, 0.30], "score": 0.8},
        {"bbox": [0.40, 0.10, 0.50, 0.30], "score": 0.7},
    ]
    cleaned, stats = clean_candidates(candidates)
    assert len(cleaned) == 2
    assert stats["after_nms"] == 2


def test_inconsistent_instance_scales_require_recovery():
    records = {
        "000001_001": {"visible": "Images/visible/000001.png", "query": "first stone pier from the left", "bbox": [.1,.1,.9,.9]},
        "000001_002": {"visible": "Images/visible/000001.png", "query": "second stone pier from the left", "bbox": [.1,.1,.9,.9]},
    }
    details = {
        "000001_001": {"candidates": [{"bbox": [.1,.1,.9,.9], "score": .9}]},
        "000001_002": {"candidates": [{"bbox": [.8,.8,.85,.85], "score": .8}]},
    }
    result = build_ordinal_proposals(records, details)
    assert result["families"][0]["needs_tiled_recovery"] is True
    assert result["change_proposals"] == []
