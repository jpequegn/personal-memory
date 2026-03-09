"""Tests for the mem CLI (click commands)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from mem.cli import cli
from mem.store import MemoryStore

_DIM = 1536


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_vec(seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(_DIM)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _patch_embed_batch(texts):
    return [_rand_vec(i) for i in range(len(texts))]


def _patch_embed(text):
    return _rand_vec(0)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "mem.db"


@pytest.fixture
def populated_store(db_path):
    """A store pre-loaded with a few chunks (no real API calls)."""
    with MemoryStore(db_path) as s:
        for i in range(5):
            s.insert(f"Chunk {i}: some text about topic {i}.",
                     _rand_vec(i),
                     source=f"source-{i % 2}",
                     fingerprint=f"fp-{i}")
    return db_path


def _cli_store(db_path: Path):
    """Context manager that patches MemoryStore to use tmp db."""
    return patch("mem.cli._store", return_value=MemoryStore(db_path))


# ---------------------------------------------------------------------------
# mem store
# ---------------------------------------------------------------------------

class TestStoreCmd:
    def test_store_inline_text(self, runner, db_path):
        with patch("mem.cli._store", return_value=MemoryStore(db_path)), \
             patch("mem.embedder.embed_batch", side_effect=_patch_embed_batch):
            result = runner.invoke(cli, ["store", "Hello world."])
        assert result.exit_code == 0
        assert "Stored" in result.output

    def test_store_from_file(self, runner, db_path, tmp_path):
        f = tmp_path / "note.txt"
        f.write_text("Content from a file.")
        with patch("mem.cli._store", return_value=MemoryStore(db_path)), \
             patch("mem.embedder.embed_batch", side_effect=_patch_embed_batch):
            result = runner.invoke(cli, ["store", "--file", str(f)])
        assert result.exit_code == 0
        assert "Stored" in result.output

    def test_store_from_stdin(self, runner, db_path):
        with patch("mem.cli._store", return_value=MemoryStore(db_path)), \
             patch("mem.embedder.embed_batch", side_effect=_patch_embed_batch):
            result = runner.invoke(cli, ["store", "--stdin"], input="stdin text")
        assert result.exit_code == 0
        assert "Stored" in result.output

    def test_store_reports_skipped_duplicates(self, runner, db_path):
        args = ["store", "Same text. Stored twice."]
        with patch("mem.cli._store", return_value=MemoryStore(db_path)), \
             patch("mem.embedder.embed_batch", side_effect=_patch_embed_batch):
            runner.invoke(cli, args)
        with patch("mem.cli._store", return_value=MemoryStore(db_path)), \
             patch("mem.embedder.embed_batch", side_effect=_patch_embed_batch):
            result = runner.invoke(cli, args)
        assert "Skipped" in result.output

    def test_store_no_input_shows_error(self, runner, db_path):
        with patch("mem.cli._store", return_value=MemoryStore(db_path)):
            result = runner.invoke(cli, ["store"])
        assert result.exit_code != 0

    def test_store_with_source_and_tags(self, runner, db_path):
        with patch("mem.cli._store", return_value=MemoryStore(db_path)), \
             patch("mem.embedder.embed_batch", side_effect=_patch_embed_batch):
            result = runner.invoke(
                cli, ["store", "tagged text", "--source", "wiki", "--tag", "ai"]
            )
        assert result.exit_code == 0
        with MemoryStore(db_path) as s:
            assert "wiki" in s.list_sources()


# ---------------------------------------------------------------------------
# mem query
# ---------------------------------------------------------------------------

class TestQueryCmd:
    def test_query_no_results(self, runner, db_path):
        with patch("mem.cli._store", return_value=MemoryStore(db_path)), \
             patch("mem.embedder.embed", return_value=_rand_vec(0)):
            result = runner.invoke(cli, ["query", "something"])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_query_returns_results(self, runner, populated_store):
        with patch("mem.cli._store", return_value=MemoryStore(populated_store)), \
             patch("mem.embedder.embed", return_value=_rand_vec(0)):
            result = runner.invoke(cli, ["query", "topic"])
        assert result.exit_code == 0
        assert "Chunk" in result.output

    def test_query_json_output(self, runner, populated_store):
        with patch("mem.cli._store", return_value=MemoryStore(populated_store)), \
             patch("mem.embedder.embed", return_value=_rand_vec(0)):
            result = runner.invoke(cli, ["query", "topic", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert all("chunk_text" in item for item in data)

    def test_query_top_k_respected(self, runner, populated_store):
        with patch("mem.cli._store", return_value=MemoryStore(populated_store)), \
             patch("mem.embedder.embed", return_value=_rand_vec(0)):
            result = runner.invoke(cli, ["query", "topic", "-k", "2", "--json"])
        data = json.loads(result.output)
        assert len(data) <= 2


# ---------------------------------------------------------------------------
# mem list
# ---------------------------------------------------------------------------

class TestListCmd:
    def test_list_empty(self, runner, db_path):
        with patch("mem.cli._store", return_value=MemoryStore(db_path)):
            result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "No memories" in result.output

    def test_list_shows_entries(self, runner, populated_store):
        with patch("mem.cli._store", return_value=MemoryStore(populated_store)):
            result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "Chunk" in result.output

    def test_list_json(self, runner, populated_store):
        with patch("mem.cli._store", return_value=MemoryStore(populated_store)):
            result = runner.invoke(cli, ["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 5

    def test_list_n_limit(self, runner, populated_store):
        with patch("mem.cli._store", return_value=MemoryStore(populated_store)):
            result = runner.invoke(cli, ["list", "-n", "2", "--json"])
        data = json.loads(result.output)
        assert len(data) == 2


# ---------------------------------------------------------------------------
# mem stats
# ---------------------------------------------------------------------------

class TestStatsCmd:
    def test_stats_empty(self, runner, db_path):
        with patch("mem.cli._store", return_value=MemoryStore(db_path)):
            result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "0" in result.output

    def test_stats_shows_count(self, runner, populated_store):
        with patch("mem.cli._store", return_value=MemoryStore(populated_store)):
            result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "5" in result.output

    def test_stats_json(self, runner, populated_store):
        with patch("mem.cli._store", return_value=MemoryStore(populated_store)):
            result = runner.invoke(cli, ["stats", "--json"])
        data = json.loads(result.output)
        assert data["total_chunks"] == 5
        assert isinstance(data["sources"], list)

    def test_stats_lists_sources(self, runner, populated_store):
        with patch("mem.cli._store", return_value=MemoryStore(populated_store)):
            result = runner.invoke(cli, ["stats"])
        assert "source-0" in result.output or "source-1" in result.output


# ---------------------------------------------------------------------------
# mem dedup
# ---------------------------------------------------------------------------

class TestDedupCmd:
    def test_dedup_no_dupes(self, runner, populated_store):
        with patch("mem.cli._store", return_value=MemoryStore(populated_store)):
            result = runner.invoke(cli, ["dedup"])
        assert result.exit_code == 0
        assert "No duplicates" in result.output

    def test_dedup_dry_run_reports(self, runner, db_path):
        with MemoryStore(db_path) as s:
            s.insert("x", _rand_vec(0), fingerprint="dup-fp")
            s.insert("x", _rand_vec(1), fingerprint="dup-fp")
        with patch("mem.cli._store", return_value=MemoryStore(db_path)):
            result = runner.invoke(cli, ["dedup", "--dry-run"])
        assert result.exit_code == 0
        assert "1 duplicated" in result.output
        assert "dup-fp"[:8] in result.output
