# tests/test_config.py
import os
from pathlib import Path
import pytest
from kiwix_rag.config import Config


def test_defaults():
    cfg = Config()
    assert cfg.db_path == Path("vector_db")
    assert cfg.embed_model == "all-MiniLM-L6-v2"
    assert cfg.ollama_url == "http://localhost:11434"
    assert cfg.llm_model == "llama3.2:3b"
    assert cfg.timeout == 300
    assert cfg.top_k == 3
    assert cfg.top_groups == 2
    assert cfg.route_threshold == pytest.approx(0.20)
    assert cfg.max_cache_size == 15
    assert cfg.max_per_group == 15
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 5000
    assert cfg.kiwix_dir is None


def test_load_yaml(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        "ollama_url: http://my-server:11434\n"
        "top_k: 10\n"
        "llm_model: phi3:mini\n"
    )
    cfg = Config.load(yaml_file)
    assert cfg.ollama_url == "http://my-server:11434"
    assert cfg.top_k == 10
    assert cfg.llm_model == "phi3:mini"
    assert cfg.embed_model == "all-MiniLM-L6-v2"  # default preserved


def test_load_missing_yaml_uses_defaults(tmp_path):
    cfg = Config.load(tmp_path / "nonexistent.yaml")
    assert cfg.top_k == 3


def test_env_override(monkeypatch, tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("top_k: 7\n")
    monkeypatch.setenv("KIWIX_RAG_TOP_K", "12")
    cfg = Config.load(yaml_file)
    assert cfg.top_k == 12  # env wins over yaml


def test_cli_override_wins_over_env(monkeypatch, tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("top_k: 7\n")
    monkeypatch.setenv("KIWIX_RAG_TOP_K", "12")
    cfg = Config.load(yaml_file, top_k=20)
    assert cfg.top_k == 20  # kwargs win over env


def test_db_path_is_pathlib(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("db_path: /tmp/mydb\n")
    cfg = Config.load(yaml_file)
    assert isinstance(cfg.db_path, Path)
    assert cfg.db_path == Path("/tmp/mydb")


def test_auto_discover_config(tmp_path, monkeypatch):
    """Config.load() with no path finds config.yaml in CWD."""
    (tmp_path / "config.yaml").write_text("top_k: 99\n")
    monkeypatch.chdir(tmp_path)
    cfg = Config.load()
    assert cfg.top_k == 99


def test_kiwix_dir_path_coercion(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("kiwix_dir: /mnt/ssd/kiwix\n")
    cfg = Config.load(yaml_file)
    assert isinstance(cfg.kiwix_dir, Path)
    assert cfg.kiwix_dir == Path("/mnt/ssd/kiwix")


def test_invalid_int_raises_valueerror(monkeypatch):
    monkeypatch.setenv("KIWIX_RAG_TOP_K", "not-a-number")
    with pytest.raises(ValueError, match="top_k"):
        Config.load()


def test_invalid_yaml_raises_valueerror(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("db_path: [\nbad yaml\n")
    with pytest.raises(ValueError, match="Invalid YAML"):
        Config.load(yaml_file)


def test_max_cache_bytes_default():
    from kiwix_rag.config import Config
    assert Config().max_cache_bytes == 11_000_000_000


def test_max_cache_bytes_override_coerces_int():
    from kiwix_rag.config import Config
    cfg = Config.load(max_cache_bytes="5000000000")
    assert cfg.max_cache_bytes == 5_000_000_000
