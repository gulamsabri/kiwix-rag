import sqlite3
from pathlib import Path
from kiwix_rag.collection_size import dir_bytes, CollectionSizer


def test_dir_bytes_sums_files(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    (tmp_path / "b.bin").write_bytes(b"y" * 250)
    assert dir_bytes(tmp_path) == 350


def test_dir_bytes_ignores_subdirs_and_missing(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.bin").write_bytes(b"z" * 999)
    assert dir_bytes(tmp_path) == 10
    assert dir_bytes(tmp_path / "does_not_exist") == 0


def _make_chroma_db(root: Path):
    con = sqlite3.connect(str(root / "chroma.sqlite3"))
    con.execute("CREATE TABLE collections (id TEXT, name TEXT)")
    con.execute("CREATE TABLE segments (id TEXT, collection TEXT)")
    con.execute("INSERT INTO collections VALUES ('cid1', 'col_a')")
    con.execute("INSERT INTO segments VALUES ('seg1', 'cid1')")
    con.execute("INSERT INTO segments VALUES ('seg2', 'cid1')")
    con.commit()
    con.close()
    for seg, n in (("seg1", 1000), ("seg2", 500)):
        d = root / seg
        d.mkdir()
        (d / "data.bin").write_bytes(b"x" * n)


def test_sizer_sums_all_segments_for_collection(tmp_path):
    _make_chroma_db(tmp_path)
    sizer = CollectionSizer(tmp_path)
    assert sizer.size("col_a") == 1500


def test_sizer_unknown_collection_is_zero(tmp_path):
    _make_chroma_db(tmp_path)
    assert CollectionSizer(tmp_path).size("missing") == 0


def test_sizer_missing_db_is_zero(tmp_path):
    # No chroma.sqlite3 at all (e.g. test/dev environment)
    assert CollectionSizer(tmp_path).size("anything") == 0


def test_sizer_corrupt_db_is_zero(tmp_path):
    (tmp_path / "chroma.sqlite3").write_bytes(b"not a database")
    assert CollectionSizer(tmp_path).size("anything") == 0
