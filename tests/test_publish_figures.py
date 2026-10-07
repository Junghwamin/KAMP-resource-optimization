"""Publication fixtures: routing only; no plotting, fitting or source-table writes."""
import copy
import csv
import json
from pathlib import Path

import pytest

from tools import analysis_snapshot
from tools import publish_figures as p


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def publication(tmp_path, monkeypatch):
    root = tmp_path / "한글 공백 프로젝트"
    legacy, rendered, snapshot = root / "outputs/figures", root / "outputs/figures_rerendered", root / "outputs/analysis_snapshot"
    for path in (legacy, rendered, root / "outputs/tables"):
        path.mkdir(parents=True)
    (root / "outputs/predictions_test_336h.csv").write_text("unchanged numeric predictions", encoding="utf-8")
    put(snapshot / "metadata.json", {"fingerprint": "fixture"})
    monkeypatch.setattr(analysis_snapshot, "load_snapshot", lambda path, **kw: {"metadata": {"fingerprint": "fixture"}})
    code = {}
    for name in ("tools/render_figures.py", "tools/figure_layout.py", "src/s00_env.py"):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("# fixture", encoding="utf-8")
        code[name] = p.digest(path)
    rows, index = [], []
    for fid in sorted(p.EXPECTED):
        name = f"{fid}_원래 그림 이름.png"
        (legacy / name).write_bytes(b"old PNG " + fid.encode())
        (root / "outputs/tables" / f"{fid}_src.csv").write_text("x,value\n1,0.12345678901234567\n", encoding="utf-8")
        (rendered / f"{fid}.png").write_bytes(b"final PNG " + fid.encode())
        row = {"fid": fid, "png": fid + ".png", "sha256": p.digest(rendered / f"{fid}.png"),
               "glyph_warnings": [], "visual_status": "pending", "overlap_candidates": []}
        put(rendered / f"{fid}.qa.json", row)
        rows.append(row)
        index.append({"fid": fid, "파일명": name, "소스표": f"{fid}_src.csv"})
    with (legacy / "figure_index.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fid", "파일명", "소스표"])
        writer.writeheader()
        writer.writerows(index)
    manifest = {"status": "rendered_pending_visual_review", "rendered": 39, "failures": [],
                "snapshot_files_unchanged": True, "raw_arrays_unchanged": True, "snapshot_fingerprint": "fixture",
                "snapshot_metadata_sha256": p.digest(snapshot / "metadata.json"),
                "renderer_sha256": {k: v for k, v in code.items() if k.startswith("tools/")},
                "source_sha256": {"s00_env.py": code["src/s00_env.py"]}, "figures": rows}
    put(rendered / "render_manifest.json", manifest)
    return root, rendered, snapshot


def test_publish_updates_all_native_paths_and_preserves_numbers_index_and_canonical_qa(publication):
    root, rendered, snapshot = publication
    before = p.protected_files(root)
    canonical = {fid: p.digest(rendered / f"{fid}.qa.json") for fid in p.EXPECTED}
    result = p.publish(root, rendered, snapshot)
    assert result["status"] == "complete" and result["published_count"] == 39
    assert p.protected_files(root) == before
    for row in result["figures"]:
        assert row["old_png_sha256"] != row["new_png_sha256"]
        assert p.digest(root / row["published_png"]) == row["new_png_sha256"]
        original = p.read_json(root / row["canonical_qa"])
        expected = copy.deepcopy(original)
        expected["png"] = Path(row["published_png"]).name
        assert p.read_json(root / row["published_qa"]) == expected
        assert p.digest(root / row["canonical_qa"]) == canonical[row["fid"]] == row["canonical_qa_sha256"]
    assert p.verify_published(root, rendered, snapshot)["status"] == "passed"


def test_republish_is_idempotent_and_retains_previous_manifest_link(publication):
    root, rendered, snapshot = publication
    p.publish(root, rendered, snapshot)
    result = p.publish(root, rendered, snapshot)
    assert result["previous_publication"]["status"] == "complete"
    assert all(r["old_png_sha256"] == r["new_png_sha256"] for r in result["figures"])
    assert p.verify_published(root, rendered, snapshot)["status"] == "passed"


@pytest.mark.parametrize("problem", ["traversal", "duplicate_fid", "wrong_fid", "missing_source"])
def test_bad_historical_index_is_rejected_before_any_png_changes(publication, problem):
    root, rendered, snapshot = publication
    index_path = root / "outputs/figures/figure_index.csv"
    with index_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    old = p.digest(root / "outputs/figures" / rows[0]["파일명"])
    if problem == "traversal":
        rows[0]["파일명"] = "../F01_escape.png"
    elif problem == "duplicate_fid":
        rows[1]["fid"] = rows[0]["fid"]
    elif problem == "wrong_fid":
        rows[0]["파일명"] = "F99_mismatch.png"
    else:
        rows[0]["소스표"] = "not_the_source.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        p.publish(root, rendered, snapshot)
    assert p.digest(root / "outputs/figures/F01_원래 그림 이름.png") == old


@pytest.mark.parametrize("problem", ["stale_snapshot", "changed_png", "missing_qa", "changed_renderer", "duplicate_render_fid"])
def test_stale_or_incomplete_render_cache_is_never_published(publication, problem):
    root, rendered, snapshot = publication
    if problem == "stale_snapshot":
        put(snapshot / "metadata.json", {"fingerprint": "changed"})
    elif problem == "changed_png":
        (rendered / "F02.png").write_bytes(b"modified")
    elif problem == "missing_qa":
        (rendered / "F02.qa.json").unlink()
    elif problem == "changed_renderer":
        (root / "tools/figure_layout.py").write_text("changed", encoding="utf-8")
    else:
        manifest = p.read_json(rendered / "render_manifest.json")
        manifest["figures"][1]["fid"] = "F01"
        put(rendered / "render_manifest.json", manifest)
    before = {f.name: p.digest(f) for f in (root / "outputs/figures").glob("*.png")}
    with pytest.raises((ValueError, FileNotFoundError)):
        p.publish(root, rendered, snapshot)
    assert {f.name: p.digest(f) for f in (root / "outputs/figures").glob("*.png")} == before


@pytest.mark.parametrize("problem", ["viewer_stale", "numeric_change", "qa_stale"])
def test_final_gate_checks_viewer_and_numeric_immutability(publication, problem):
    root, rendered, snapshot = publication
    result = p.publish(root, rendered, snapshot)
    if problem == "numeric_change":
        (root / "outputs/tables/F01_src.csv").write_text("changed values", encoding="utf-8")
    elif problem == "qa_stale":
        (root / result["figures"][0]["published_qa"]).write_text("{}", encoding="utf-8")
    else:
        (root / result["figures"][0]["published_png"]).write_bytes(b"old cached PNG")
    with pytest.raises(ValueError):
        p.verify_published(root, rendered, snapshot)
