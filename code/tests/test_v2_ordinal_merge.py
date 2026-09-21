from scripts.merge_v2_ordinal_tiled import inject_tiled_candidates


def test_tiled_candidates_are_shared_by_family_queries():
    details = {"000001_001": {"reranked_candidates": [{"bbox": [0, 0, .1, .1]}]}}
    tiled = {
        "family": {
            "query_ids": ["000001_001", "000001_002"],
            "candidates": [{"bbox": [.2, .2, .3, .3], "score": .8}],
        }
    }
    merged, injected = inject_tiled_candidates(details, tiled)
    assert len(merged["000001_001"]["reranked_candidates"]) == 2
    assert len(merged["000001_002"]["reranked_candidates"]) == 1
    assert injected == 2
