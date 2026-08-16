from pisteflow.sources.oman import _record_count


def test_record_count_uses_endpoint_native_lists() -> None:
    assert _record_count("athletes", {"athletes": [{}, {}], "count": 99}) == 2
    assert _record_count("rankings", {"rankings": [{}]}) == 1
    assert _record_count("birthyears", {"byYear": [{}, {}], "total": 3}) == 3
    assert _record_count("stats", {"ok": True}) == 1

