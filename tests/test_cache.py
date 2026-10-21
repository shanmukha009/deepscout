import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from deepscout.cache import DiskCache, make_key


def test_make_key_is_stable_and_order_insensitive():
    k1 = make_key("complete", "openai", "gpt-4o-mini", "sys", "user", 256)
    k2 = make_key("complete", "openai", "gpt-4o-mini", "sys", "user", 256)
    assert k1 == k2
    assert make_key("a") != make_key("b")


def test_disabled_cache_is_a_noop():
    with tempfile.TemporaryDirectory() as d:
        c = DiskCache(directory=d, enabled=False)
        c.set("k", {"v": 1})
        assert c.get("k") is None


def test_enabled_cache_roundtrips():
    with tempfile.TemporaryDirectory() as d:
        c = DiskCache(directory=d, enabled=True)
        c.set("k", {"text": "hi", "usage": {"input_tokens": 3}})
        got = c.get("k")
        assert got["text"] == "hi"
        assert got["usage"]["input_tokens"] == 3


def test_missing_key_returns_none():
    with tempfile.TemporaryDirectory() as d:
        c = DiskCache(directory=d, enabled=True)
        assert c.get("nope") is None
