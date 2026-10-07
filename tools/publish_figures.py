"""Publish verified final native figures into the existing viewer filenames.

PNG bytes are copied and QA changes only its png filename to the viewer name.
Numeric CSVs, source code, the historical index and core-run evidence are preserved.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.verify_submission import confined, digest, hashed_files, read_json, require, write_json  # noqa: E402

EXPECTED = {f"F{i:02}" for i in range(1, 40)}


def protected_files(root):
    root = Path(root)
    candidates = [root / "outputs/figures/figure_index.csv", root / "outputs/predictions_test_336h.csv"]
    candidates += sorted((root / "outputs/tables").glob("*.csv"))
    require(all(p.is_file() for p in candidates[:2]), "Required historical figure index or prediction CSV is missing")
    require(len(candidates) >= 41, "Expected numeric source tables are missing")
    return {p.relative_to(root).as_posix(): digest(p) for p in candidates}


def read_targets(root):
    index_path = Path(root) / "outputs/figures/figure_index.csv"
    with index_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 39 and {r.get("fid") for r in rows} == EXPECTED, "Historical index needs exactly 39 unique FIDs")
    names = []
    result = {}
    for row in rows:
        fid, name = row["fid"], row.get("파일명", "")
        require(name and "/" not in name and "\\" not in name and Path(name).name == name
                and name.startswith(fid + "_") and name.lower().endswith(".png"), f"Unsafe or FID-mismatched historical filename: {fid}")
        require(name.casefold() not in names, "Duplicate historical target filename")
        names.append(name.casefold())
        table = row.get("소스표")
        require(table == f"{fid}_src.csv" and (Path(root) / "outputs/tables" / table).is_file(), f"Missing or mismatched numeric source table: {fid}")
        result[fid] = name
    return result


def validate_rendered(root, rendered, snapshot):
    root, rendered, snapshot = Path(root), Path(rendered), Path(snapshot)
    from tools.analysis_snapshot import load_snapshot
    meta = load_snapshot(snapshot, verify=True)["metadata"]
    manifest = read_json(rendered / "render_manifest.json")
    require(manifest.get("status") == "rendered_pending_visual_review" and manifest.get("rendered") == 39
            and not manifest.get("failures"), "Baseline rendering is incomplete")
    require(manifest.get("snapshot_files_unchanged") is True and manifest.get("raw_arrays_unchanged") is True,
            "Renderer changed numeric snapshot data")
    require(manifest.get("snapshot_metadata_sha256") == digest(snapshot / "metadata.json")
            and manifest.get("snapshot_fingerprint") == meta.get("fingerprint"), "Rendered figures belong to another snapshot")
    require(set(manifest.get("renderer_sha256", {})) == {"tools/render_figures.py", "tools/figure_layout.py"}, "Incomplete renderer provenance")
    hashed_files(root, manifest["renderer_sha256"])
    source = {f"src/{key}": value for key, value in manifest.get("source_sha256", {}).items()}
    hashed_files(root, source)
    require(set(source) == {p.relative_to(root).as_posix() for p in (root / "src").glob("s0*.py")}, "Rendering source coverage mismatch")
    figures = manifest.get("figures", [])
    require(len(figures) == 39 and {f.get("fid") for f in figures} == EXPECTED, "Rendering has missing/duplicate FIDs")
    for row in figures:
        fid = row["fid"]
        require(row.get("png") == fid + ".png", f"Unexpected canonical filename: {fid}")
        png = confined(rendered, row["png"])
        require(png.is_file() and digest(png) == row.get("sha256"), f"Canonical PNG hash mismatch: {fid}")
        qa = read_json(rendered / f"{fid}.qa.json")
        require(qa.get("fid") == fid and qa.get("sha256") == row["sha256"] and not qa.get("glyph_warnings"),
                f"Canonical QA missing or inconsistent: {fid}")
    return manifest


def _copy_atomic(source, destination):
    temp = destination.with_suffix(destination.suffix + ".publish.tmp")
    temp.write_bytes(source.read_bytes())
    temp.replace(destination)


def mapped_qa_bytes(source, target_png_name):
    qa = read_json(source)
    qa["png"] = target_png_name
    return (json.dumps(qa, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def publish(root, rendered, snapshot):
    root, rendered, snapshot = Path(root).resolve(), Path(rendered).resolve(), Path(snapshot).resolve()
    require(rendered.is_relative_to(root) and snapshot.is_relative_to(root), "Publishing inputs must be inside this submission project")
    targets = read_targets(root)
    source_manifest = validate_rendered(root, rendered, snapshot)
    before = protected_files(root)
    destination = root / "outputs/figures"
    entries = []
    for row in sorted(source_manifest["figures"], key=lambda r: r["fid"]):
        fid = row["fid"]
        old_png = confined(destination, targets[fid])
        old_qa = old_png.with_suffix(".qa.json")
        require(old_png.is_file(), f"Historical target figure is missing: {fid}")
        entries.append({"fid": fid, "canonical_png": (rendered / f"{fid}.png").relative_to(root).as_posix(),
                        "canonical_qa": (rendered / f"{fid}.qa.json").relative_to(root).as_posix(),
                        "published_png": old_png.relative_to(root).as_posix(), "published_qa": old_qa.relative_to(root).as_posix(),
                        "old_png_sha256": digest(old_png), "new_png_sha256": row["sha256"],
                        "old_qa_sha256": digest(old_qa) if old_qa.is_file() else None,
                        "canonical_qa_sha256": digest(rendered / f"{fid}.qa.json"),
                        "new_qa_sha256": hashlib.sha256(mapped_qa_bytes(rendered / f"{fid}.qa.json", old_png.name)).hexdigest()})
    path = root / "outputs/verification/published_figures.json"
    previous = read_json(path) if path.is_file() else None
    manifest = {"schema": "kamp.published_figures.v1", "status": "publishing",
                "published_at_utc": datetime.now(timezone.utc).isoformat(),
                "render_manifest_path": (rendered / "render_manifest.json").relative_to(root).as_posix(),
                "render_manifest_sha256": digest(rendered / "render_manifest.json"),
                "snapshot_metadata_sha256": source_manifest["snapshot_metadata_sha256"],
                "snapshot_fingerprint": source_manifest["snapshot_fingerprint"],
                "renderer_sha256": source_manifest["renderer_sha256"], "source_sha256": source_manifest["source_sha256"],
                "protected_files_sha256": before, "figures": entries,
                "qa_copy_policy": "Canonical QA file preserved unchanged; published QA changes only png to the existing viewer basename. Both canonical and mapped QA hashes are recorded.",
                "previous_publication": ({"status": previous.get("status"), "published_at_utc": previous.get("published_at_utc"),
                                          "render_manifest_sha256": previous.get("render_manifest_sha256")} if previous else None)}
    write_json(path, manifest)
    try:
        for entry in entries:
            for kind in ("png", "qa"):
                src, dst = root / entry["canonical_" + kind], root / entry["published_" + kind]
                if kind == "png":
                    _copy_atomic(src, dst)
                else:
                    tmp = dst.with_suffix(dst.suffix + ".publish.tmp")
                    tmp.write_bytes(mapped_qa_bytes(src, Path(entry["published_png"]).name))
                    tmp.replace(dst)
                require(digest(dst) == entry[f"new_{kind}_sha256"], f"Publish verification failed: {entry['fid']}/{kind}")
        require(before == protected_files(root), "Numeric CSV or historical figure index changed during publication")
        manifest.update(status="complete", published_count=len(entries), protected_files_unchanged=True)
        write_json(path, manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc).replace(str(root), "<project>")})
        write_json(path, manifest)
        raise


def verify_published(root, rendered, snapshot):
    root, rendered, snapshot = Path(root), Path(rendered), Path(snapshot)
    path = root / "outputs/verification/published_figures.json"
    manifest = read_json(path)
    require(manifest.get("schema") == "kamp.published_figures.v1" and manifest.get("status") == "complete"
            and manifest.get("published_count") == 39 and manifest.get("protected_files_unchanged") is True,
            "Viewer figure publication is missing or incomplete")
    source = read_json(rendered / "render_manifest.json")
    require(manifest.get("render_manifest_sha256") == digest(rendered / "render_manifest.json")
            and manifest.get("snapshot_metadata_sha256") == digest(snapshot / "metadata.json")
            and manifest.get("snapshot_fingerprint") == read_json(snapshot / "metadata.json").get("fingerprint"),
            "Viewer publication belongs to stale rendering/snapshot")
    require(manifest.get("protected_files_sha256") == protected_files(root), "Protected numeric CSV/index changed after publication")
    targets = read_targets(root)
    entries = manifest.get("figures", [])
    require(len(entries) == 39 and {r.get("fid") for r in entries} == EXPECTED, "Published FID coverage mismatch")
    sources = {r["fid"]: r["sha256"] for r in source["figures"]}
    for row in entries:
        fid = row["fid"]
        require(row.get("published_png") == "outputs/figures/" + targets[fid], "Publication mapping differs from historical index")
        require(row.get("published_qa") == str(Path(row["published_png"]).with_suffix(".qa.json")).replace("\\", "/"),
                "Published QA mapping differs from its native PNG")
        require(row.get("canonical_png") == (rendered / f"{fid}.png").relative_to(root).as_posix()
                and row.get("canonical_qa") == (rendered / f"{fid}.qa.json").relative_to(root).as_posix(), "Canonical publication source mismatch")
        require(row.get("new_png_sha256") == sources[fid] == digest(rendered / f"{fid}.png"), f"Canonical final PNG mismatch: {fid}")
        for kind in ("png", "qa"):
            target = confined(root, row["published_" + kind])
            canonical = confined(root, row["canonical_" + kind])
            expected = digest(canonical) if kind == "png" else hashlib.sha256(mapped_qa_bytes(canonical, targets[fid])).hexdigest()
            if kind == "qa":
                require(digest(canonical) == row.get("canonical_qa_sha256"), f"Canonical QA changed after publication: {fid}")
            require(target.is_file() and digest(target) == expected == row.get(f"new_{kind}_sha256"),
                    f"Default viewer still points to an old PNG/QA: {fid}/{kind}")
    return {"status": "passed", "published_count": 39, "publication_manifest_sha256": digest(path),
            "protected_files_unchanged": True, "scope": "default viewer legacy filenames equal final native render"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered", type=Path, default=ROOT / "outputs/figures_rerendered")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "outputs/analysis_snapshot")
    args = parser.parse_args(argv)
    publish(ROOT, args.rendered, args.snapshot)
    print("PUBLISHED_39_FIGURES numeric_csv_and_historical_index_unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
