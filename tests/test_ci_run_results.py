"""Tests for the CI run_results.json freshness guard.

Regression for algolia/data#4811: dbt seed and dbt run both died at Databricks
auth before executing a node, neither wrote target/run_results.json, and the CI
handler read the file left behind by an earlier unrelated dbt invocation —
reporting a seed "loaded successfully" and an unrelated model as built.
"""

import json
from pathlib import Path

from jirade.mcp.handlers.dbt_diff import (
    _clear_run_results,
    _not_succeeded,
    _parse_run_results,
)


def _write_results(project: Path, results: list[dict]) -> Path:
    target = project / "target"
    target.mkdir(parents=True, exist_ok=True)
    path = target / "run_results.json"
    path.write_text(json.dumps({"results": results}))
    return path


def _result(unique_id: str, status: str) -> dict:
    return {"unique_id": unique_id, "status": status}


class TestClearRunResults:
    def test_removes_stale_file(self, tmp_path: Path):
        path = _write_results(tmp_path, [_result("model.p.stale_model", "success")])
        assert path.exists()

        returned = _clear_run_results(tmp_path)

        assert returned == path
        assert not path.exists()

    def test_noop_when_absent(self, tmp_path: Path):
        returned = _clear_run_results(tmp_path)

        assert returned == tmp_path / "target" / "run_results.json"
        assert not returned.exists()


class TestParseRunResults:
    def test_none_when_file_missing(self, tmp_path: Path):
        assert _parse_run_results(tmp_path, "model.") is None

    def test_none_when_unreadable(self, tmp_path: Path):
        target = tmp_path / "target"
        target.mkdir()
        (target / "run_results.json").write_text("{not json")

        assert _parse_run_results(tmp_path, "seed.") is None

    def test_empty_results_is_not_none(self, tmp_path: Path):
        _write_results(tmp_path, [])

        assert _parse_run_results(tmp_path, "model.") == ([], [])

    def test_splits_by_status_and_resource_type(self, tmp_path: Path):
        _write_results(
            tmp_path,
            [
                _result("model.p.built_ok", "success"),
                _result("model.p.built_pass", "pass"),
                _result("model.p.broken", "error"),
                _result("model.p.skipped_one", "skipped"),
                _result("seed.p.my_seed", "success"),
                _result("seed.p.bad_seed", "error"),
            ],
        )

        assert _parse_run_results(tmp_path, "model.") == (
            ["built_ok", "built_pass"],
            ["broken"],
        )
        assert _parse_run_results(tmp_path, "seed.") == (["my_seed"], ["bad_seed"])


class TestNotSucceeded:
    def test_all_flagged_when_nothing_ran(self):
        assert _not_succeeded(["a", "b"], None) == ["a", "b"]

    def test_all_flagged_when_selection_was_empty(self):
        assert _not_succeeded(["a", "b"], ([], [])) == ["a", "b"]

    def test_flags_missing_and_errored_only(self):
        parsed = (["loaded"], ["errored"])

        assert _not_succeeded(["loaded", "errored", "never_ran"], parsed) == ["errored", "never_ran"]

    def test_empty_when_all_loaded(self):
        assert _not_succeeded(["a"], (["a", "other"], [])) == []


class TestStaleResultsRegression:
    """The #4811 shape end to end at the helper level."""

    def test_stale_model_results_do_not_count_as_a_seed_load(self, tmp_path: Path):
        # An earlier, unrelated dbt run left this behind.
        _write_results(tmp_path, [_result("model.p.rpt_transformations_executed", "success")])

        # CI clears before invoking dbt; dbt then dies at auth and writes nothing.
        _clear_run_results(tmp_path)
        parsed = _parse_run_results(tmp_path, "seed.")

        assert parsed is None
        assert _not_succeeded(["seed_reference_metis_compatibility_apps"], parsed) == [
            "seed_reference_metis_compatibility_apps"
        ]

    def test_stale_model_results_do_not_count_as_built_models(self, tmp_path: Path):
        _write_results(tmp_path, [_result("model.p.rpt_transformations_executed", "success")])

        _clear_run_results(tmp_path)

        assert _parse_run_results(tmp_path, "model.") is None
