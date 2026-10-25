from unittest.mock import MagicMock

import pytest

from vpe import http
from vpe.http import CachedSession


def _resp(code: int, text: str = "{}") -> MagicMock:
    r = MagicMock()
    r.status_code, r.text = code, text
    return r


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)


def test_cache_hit_avoids_network(tmp_path):
    s = CachedSession(cache_dir=tmp_path)
    s._session.get = MagicMock(return_value=_resp(200, '{"a": 1}'))
    assert s.get_json("https://example.org/x", {"q": 1}) == {"a": 1}
    assert s.get_json("https://example.org/x", {"q": 1}) == {"a": 1}
    assert s._session.get.call_count == 1
    # different params -> different cache entry
    s.get_json("https://example.org/x", {"q": 2})
    assert s._session.get.call_count == 2


def test_404_is_cached_as_none(tmp_path):
    s = CachedSession(cache_dir=tmp_path)
    s._session.get = MagicMock(return_value=_resp(404))
    assert s.get_text("https://example.org/missing") is None
    assert s.get_text("https://example.org/missing") is None
    assert s._session.get.call_count == 1


def test_retries_then_succeeds(tmp_path):
    s = CachedSession(cache_dir=tmp_path, max_retries=3)
    s._session.get = MagicMock(side_effect=[_resp(503), _resp(429), _resp(200, "ok")])
    assert s.get_text("https://example.org/flaky") == "ok"
    assert s._session.get.call_count == 3


def test_gives_up_after_retries(tmp_path):
    s = CachedSession(cache_dir=tmp_path, max_retries=1)
    s._session.get = MagicMock(return_value=_resp(500))
    with pytest.raises(RuntimeError):
        s.get_text("https://example.org/down")


def test_offline_mode_without_cache_raises(tmp_path):
    with pytest.raises(RuntimeError):
        CachedSession(cache_dir=tmp_path, offline=True).get_text("https://example.org/x")


def test_ncbi_params_added_and_api_key_not_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("NCBI_API_KEY", "secret")
    s = CachedSession(cache_dir=tmp_path)
    s._session.get = MagicMock(return_value=_resp(200, "{}"))
    s.get_text("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", {"db": "clinvar"})
    sent = s._session.get.call_args.kwargs["params"]
    assert sent["tool"] == "variant-pathogenicity-explorer" and sent["api_key"] == "secret"
    for f in tmp_path.rglob("*.cache"):
        assert "secret" not in f.read_text()
