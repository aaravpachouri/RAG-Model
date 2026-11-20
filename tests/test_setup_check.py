"""Setup-checker logic tests - no Ollama server required."""

from setup_check import (
    format_settings,
    is_vision_capable,
    needs_newer_ollama,
    parse_version,
    recommend,
)


def models(*names):
    return [{"name": n, "size": None} for n in names]


def test_parse_version():
    assert parse_version("0.12.7") == (0, 12, 7)
    assert parse_version("0.40.16") > (0, 12, 7)
    assert parse_version("junk") is None


def test_vision_detection():
    assert is_vision_capable("qwen3-vl:4b-instruct")
    assert is_vision_capable("qwen2.5vl:3b")
    assert is_vision_capable("llama3.2-vision:11b")
    assert not is_vision_capable("qwen2.5:3b")
    assert not is_vision_capable("llama3.2:3b")


def test_qwen3_vl_version_gate():
    assert needs_newer_ollama("qwen3-vl:2b", (0, 12, 6))
    assert not needs_newer_ollama("qwen3-vl:2b", (0, 12, 7))
    assert not needs_newer_ollama("qwen2.5vl:3b", (0, 5, 0))


def test_recommend_prefers_one_qwen_for_both_roles():
    result = recommend(
        models("qwen3-vl:4b-instruct", "llama3.2:3b"),
        (0, 40, 0),
    )
    assert result["generation"] == "qwen3-vl:4b-instruct"
    assert result["vision"] == "qwen3-vl:4b-instruct"
    assert result["settings"] == {
        "RAG_MODEL": "qwen3-vl:4b-instruct",
        "RAG_VISION_MODEL": "qwen3-vl:4b-instruct",
    }


def test_recommend_text_only_install():
    result = recommend(models("qwen2.5:3b"), (0, 40, 0))
    assert result["generation"] == "qwen2.5:3b"
    assert result["vision"] is None
    assert "RAG_VISION_MODEL" not in result["settings"]


def test_recommend_flags_old_ollama():
    result = recommend(models("qwen3-vl:2b"), (0, 12, 6))
    assert result["vision"] is None
    assert any("0.12.7" in note for note in result["notes"])


def test_recommend_empty():
    result = recommend([], (0, 40, 0))
    assert result["generation"] is None
    assert result["settings"] == {}
    assert result["notes"]


def test_format_settings_both_shells():
    out = format_settings({"RAG_MODEL": "qwen3-vl:2b"})
    assert "export RAG_MODEL=qwen3-vl:2b" in out
    assert '$env:RAG_MODEL="qwen3-vl:2b"' in out
