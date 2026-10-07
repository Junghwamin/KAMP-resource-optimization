"""Strict local verification: computation and recorded visual inspection are separate.

This checker never trains models or infers a visual pass from overlap counts.
FULL_OK covers local code/results/figures only; PDF and KAMP execution are separate.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as package_metadata
import json
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENTS = ("bootstrap", "diagnostics", "sensitivity", "controlled", "temporal", "policy")
FIGURE_IDS = tuple([f"F{i:02}" for i in range(1, 40)] + [f"E{i:02}" for i in range(1, 7)])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def confined(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Artifact path escapes its directory: {relative}")
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hashed_files(root, hashes):
    require(isinstance(hashes, dict) and bool(hashes), "Artifact hash manifest is empty")
    for relative, expected in hashes.items():
        path = confined(root, relative)
        require(path.is_file() and digest(path) == expected, f"Artifact hash mismatch: {relative}")


def _load_snapshot(path):
    from tools.analysis_snapshot import load_snapshot
    return load_snapshot(path, verify=True)


def _fit_relevant(relative):
    suffix = Path(relative).suffix
    return ((relative.startswith("src/") and suffix == ".py")
            or (relative.startswith("data/") and suffix == ".csv")
            or (relative.startswith("serving/") and suffix == ".py")
            or (relative.startswith("serving/config/") and suffix == ".json") or relative in {
        "run_all.py", "requirements.txt", "tools/isolated_regime_cv.py",
        "tools/model_persistence.py", "tools/build_serving_core.py"})


def verify_core_provenance(root, index, snapshot_sources):
    """Keep original run provenance immutable; bind actual fit paths strictly."""
    provenance = index.get("provenance", {})
    original = provenance.get("files", {})
    require(bool(original), "Core-start source provenance is missing")
    text = "".join(sorted(f"{value}  {name}\n" for name, value in original.items()))
    require(hashlib.sha256(text.encode("utf-8")).hexdigest() == provenance.get("code_sha256"),
            "Core-start provenance aggregate is inconsistent")
    fixed = {name: value for name, value in original.items() if _fit_relevant(name)}
    require("run_all.py" in fixed and any(p.startswith("data/") for p in fixed), "Core-start provenance lacks numeric execution/data identity")
    hashed_files(root, fixed)
    current_files = {p.relative_to(root).as_posix() for folder in ("src", "data", "serving")
                     for p in (Path(root) / folder).rglob("*") if p.is_file()
                     and _fit_relevant(p.relative_to(root).as_posix()) and "__pycache__" not in p.parts}
    require(current_files <= set(fixed), "New fit/data/serving file was added after core execution began")
    for name, value in snapshot_sources.items():
        if _fit_relevant(name):
            require(original.get(name) == value, f"Core-start and snapshot source identity differ: {name}")
    return {"status": "passed", "immutable_fit_file_count": len(fixed),
            "snapshot_cross_checked": sorted(set(fixed) & set(snapshot_sources)),
            "other_code_policy": "Current helpers/tests/viewers/docs require separate current-code full-suite evidence"}


def verify_core(root, snapshot, *, compare_runtime=True):
    """Verify completed 62-step execution and its exact exported numeric snapshot."""
    root, snapshot = Path(root), Path(snapshot)
    import run_all
    index = read_json(root / "outputs/steps/index.json")
    require(index.get("status") == "ok" and index.get("mode") == "FULL", "Core execution is not completed FULL")
    expected = [s.split(":")[0] for s in run_all.EXPECTED_STEPS] + [s[0] for s in run_all.EXTRA_STEPS]
    steps = index.get("steps", [])
    require(len(expected) == 62 and [s.get("id") for s in steps] == expected, "Core 62-step identity/order mismatch")
    require(all(s.get("status") == "ok" for s in steps), "A required core/serving/test/check step was skipped or failed")
    flags = index.get("flags", {})
    require(not any(flags.get(k) for k in ("fast", "skip_serving", "skip_bundle", "no_tests")), "Core used prohibited skip/FAST flags")
    serving = index.get("serving", {})
    require(all(serving.get(s, {}).get("status") == "ok" for s in ("S.1", "S.2", "S.3", "S.4")), "Serving/S4 test evidence incomplete")
    checks = index.get("checks", {}).get("hard", [])
    require(checks and all(item.get("ok") is True for item in checks), "Core hard completion checks incomplete")
    for step in steps:
        require(step.get("log") and confined(root, step["log"]).is_file(), f"Missing core step log: {step.get('id')}")
    ns = _load_snapshot(snapshot)
    meta = ns["metadata"]
    require(index.get("analysis_export", {}).get("status") == "complete", "Core did not verify its requested analysis export")
    require(index["analysis_export"].get("fingerprint") == meta.get("fingerprint"), "Core index and snapshot refer to different runs")
    require(meta.get("run_metadata", {}).get("mode") == "FULL", "Snapshot is not a FULL core export")
    require(meta["run_metadata"].get("run_started") == index.get("started"), "Snapshot start identity differs from core run")
    sources = meta.get("source_hashes", {})
    require(sources and any(p.startswith("data/") for p in sources), "Snapshot lacks source/data provenance")
    # Conservatively reject changed source/data; figure-only rendering itself can
    # still use old immutable snapshots, but a new full submission cannot silently
    # claim that an old fit verifies modified scientific source.
    # Snapshot records all source-directory files, while historical run_all
    # provenance covers executable stages, not their README. Documentation is
    # bound to the current-code full-suite manifest instead of the earlier fit.
    hashed_files(root, {name: value for name, value in sources.items()
                       if _fit_relevant(name) or name == "experiments/config.json"})
    provenance = verify_core_provenance(root, index, sources)
    runtime = meta.get("runtime", {})
    require(runtime.get("python") and runtime.get("versions"), "Snapshot lacks runtime provenance")
    if compare_runtime:
        require(platform.python_version() == runtime["python"], "Core reuse Python version differs from recorded runtime")
        differences = {}
        for package, recorded in runtime["versions"].items():
            if recorded == "unavailable":
                continue
            try:
                current = package_metadata.version(package)
            except package_metadata.PackageNotFoundError:
                current = "unavailable"
            if current != recorded:
                differences[package] = {"recorded": recorded, "current": current}
        require(not differences, f"Core reuse/runtime package mismatch: {differences}")
    return {"status": "passed", "step_count": len(steps), "S4": serving["S.4"],
            "snapshot_fingerprint": meta["fingerprint"], "snapshot_metadata_sha256": digest(snapshot / "metadata.json"),
            "core_index_sha256": digest(root / "outputs/steps/index.json"),
            "source_provenance": provenance,
            "recorded_runtime": runtime, "current_platform_note": "Saved platform provenance does not establish a new KAMP execution"}


def verify_experiments(root, snapshot, run_dir, config):
    import run_experiments
    require(Path(root).resolve() == run_experiments.ROOT.resolve(), "Experiment runner root differs from verifier root")
    outputs = {}
    for name in EXPERIMENTS:
        folder = Path(run_dir) / name
        manifest = read_json(folder / "executor_manifest.json")
        require(manifest.get("status") == "complete", f"Experiment incomplete: {name}")
        require(manifest.get("experiment") == name, f"Experiment identity mismatch: {name}")
        key, _ = run_experiments.fingerprint(name, config, Path(snapshot))
        require(manifest.get("fingerprint") == key, f"Experiment input/source/config/runtime fingerprint mismatch: {name}")
        require(manifest.get("inputs", {}).get("config") == config, f"Experiment configuration mismatch: {name}")
        require(manifest.get("summary", {}).get("status") == "complete", f"Experiment summary incomplete: {name}")
        require("summary.json" in manifest.get("files_sha256", {}), f"Experiment summary hash missing: {name}")
        hashed_files(folder, manifest["files_sha256"])
        require(read_json(folder / "summary.json").get("status") == "complete", f"Saved summary incomplete: {name}")
        outputs[name] = {"status": "passed", "fingerprint": key, "executor_manifest_sha256": digest(folder / "executor_manifest.json"),
                         "artifact_count": len(manifest["files_sha256"])}
    status = read_json(Path(run_dir) / "run_status.json")
    require(status.get("status") == "complete" and status.get("all_six_verified") is True, "Six-experiment aggregate is incomplete")
    return {"status": "passed", "experiments": outputs}


def verify_quality(run_dir, config):
    """Check actual scientific flags and arithmetic, without demanding good scores."""
    import numpy as np
    import pandas as pd
    run_dir = Path(run_dir)
    summaries = {name: read_json(run_dir / name / "summary.json") for name in EXPERIMENTS}
    require(summaries["bootstrap"].get("one_day_reference_verified") is True, "One-day bootstrap reference was not verified")
    for name in ("controlled", "temporal"):
        summary = summaries[name]
        require(summary.get("completed_outer_count") == 4 and not summary.get("failures"), f"Not all four outer windows completed: {name}")
        require(summary.get("protocol", {}).get("full_protocol_guard", {}).get("matches_registered_full") is True,
                f"Scientific full-protocol guard failed: {name}")
    sensitivity = summaries["sensitivity"]
    require(set(sensitivity.get("scenario_ids", [])) == {s["id"] for s in config["sensitivity"]["scenarios"]},
            "Input-sensitivity scenario coverage differs from protocol")
    delta = sensitivity.get("c0_max_absolute_difference", {})
    require(delta and all(np.isfinite(x) and x <= config["sensitivity"].get("atol", 1e-8) for x in delta.values()),
            "Frozen C0 reference difference exceeds declared tolerance")
    audit = pd.read_csv(run_dir / "sensitivity/feature_consistency_audit.csv")
    for name in ("historical_power_features_unchanged", "headcount_zero_support_unchanged",
                 "production_daily_total_rule_passed", "shutdown_unchanged"):
        require(len(audit) and audit[name].astype(str).str.lower().isin(["true", "1"]).all(), f"Input-feature invariance failed: {name}")
    policy = summaries["policy"]
    require(policy.get("constraint_audit_passed") is True and policy.get("outer_count") == 4, "Policy constraint/outer coverage incomplete")
    constraints = pd.read_csv(run_dir / "policy/policy_constraint_audit.csv")
    require(len(constraints) and constraints.passed.astype(str).str.lower().isin(["true", "1"]).all(), "Policy constraint row failed")
    rates = pd.read_csv(run_dir / "diagnostics/condition_confusion_rates.csv")
    require(len(rates) and ((rates.TP + rates.FN == rates.P) & (rates.FP + rates.TN == rates.N)
                          & (rates.P + rates.N == rates.n_hours)).all(), "Conditional confusion arithmetic mismatch")
    temporal = pd.read_csv(run_dir / "temporal/temporal_outer_predictions.csv")
    metric = pd.read_csv(run_dir / "temporal/temporal_outer_metrics.csv")
    usable = temporal.evaluation_usable.astype(str).str.lower()
    require(usable.isin(["true", "false", "1", "0"]).all(), "Invalid temporal quality flag")
    temporal = temporal.loc[usable.isin(["true", "1"])]
    for row in metric.itertuples():
        part = temporal[(temporal.outer == row.outer) & (temporal.model == row.model)]
        require(len(part) == row.n and row.TP + row.FN + row.FP + row.TN == row.n, "Temporal quality-mask counts differ from metrics")
        require(np.isclose(np.abs(part.y_avg - part.pred_avg).mean(), row.MAE, atol=1e-8, rtol=1e-8), "Temporal MAE does not reproduce from usable rows")
    return {"status": "passed", "condition_rows": len(rates), "temporal_metric_rows": len(metric),
            "policy_partial_observation": policy.get("partial_observation"),
            "note": "Negative improvements, non-significance and incomplete-observation disclosures are valid outcomes"}


def verify_figures(figures, run_dir, *, snapshot, root=ROOT):
    figures, run_dir = Path(figures), Path(run_dir)
    rows = []
    for name, ids in (("render_manifest.json", FIGURE_IDS[:39]), ("experiment_render_manifest.json", FIGURE_IDS[39:])):
        manifest = read_json(figures / name)
        require(manifest.get("status") == "rendered_pending_visual_review", f"Figure renderer is incomplete/failed: {name}")
        require(not manifest.get("failures") and manifest.get("rendered") == len(ids), f"Figure count/failure mismatch: {name}")
        items = manifest.get("figures", [])
        require(len(items) == len(ids) and {r.get("fid") for r in items} == set(ids), f"Missing or duplicate figure IDs: {name}")
        if name == "render_manifest.json":
            require(manifest.get("snapshot_files_unchanged") is True and manifest.get("raw_arrays_unchanged") is True,
                    "Figure rendering changed snapshot/numeric inputs")
            meta = read_json(Path(snapshot) / "metadata.json")
            require(manifest.get("snapshot_metadata_sha256") == digest(Path(snapshot) / "metadata.json")
                    and manifest.get("snapshot_fingerprint") == meta.get("fingerprint"),
                    "Baseline figures belong to a different snapshot")
            source_hashes = {f"src/{key}": value for key, value in manifest.get("source_sha256", {}).items()}
            hashed_files(root, source_hashes)
            require(set(source_hashes) == {p.relative_to(root).as_posix() for p in (Path(root) / "src").glob("s0*.py")},
                    "Figure source manifest does not cover current rendering stages")
        require(set(manifest.get("renderer_sha256", {})) == {"tools/render_figures.py", "tools/figure_layout.py"},
                "Renderer provenance must identify the renderer and layout helper")
        hashed_files(root, manifest["renderer_sha256"])
        for item in items:
            fid = item["fid"]
            png = confined(figures, item["png"])
            require(png.is_file() and digest(png) == item.get("sha256"), f"Rendered PNG hash mismatch: {fid}")
            qa = read_json(figures / f"{fid}.qa.json")
            require(qa.get("fid") == fid and qa.get("sha256") == item.get("sha256"), f"Figure QA linkage mismatch: {fid}")
            require(not qa.get("glyph_warnings"), f"Missing glyph warnings: {fid}")
            require((figures / f"{fid}_src.csv").is_file(), f"Figure source table absent: {fid}")
            if fid.startswith("E"):
                matches = list(run_dir.rglob(str(item.get("source_table", "__missing__"))))
                require(len(matches) == 1 and digest(matches[0]) == item.get("source_sha256"), f"Experiment figure source hash mismatch: {fid}")
            variant = qa.get("report_variant")
            if variant:
                variant_file = confined(figures, variant["path"])
                require(variant_file.is_file() and digest(variant_file) == variant.get("sha256"), f"Report variant PNG hash mismatch: {fid}")
                vqa = read_json(variant_file.with_suffix(".qa.json"))
                require(vqa.get("sha256") == variant["sha256"] and not vqa.get("glyph_warnings")
                        and not vqa.get("outside_canvas_text") and vqa.get("readability_pass") is True
                        and float(vqa.get("minimum_effective_font_pt", 0)) >= 8,
                        f"Report variant readability/QA failed: {fid}")
            rows.append({"fid": fid, "png_sha256": item["sha256"], "report_variant": variant,
                         "overlap_candidate_count": len(qa.get("overlap_candidates", [])),
                         "minimum_font_pt": qa.get("minimum_font_pt"), "visual_status": "pending"})
    known_variant_paths = {r["report_variant"]["path"] for r in rows if r.get("report_variant")}
    extras = []
    for path in sorted((figures / "report_variants").glob("*.png")):
        relative = path.relative_to(figures).as_posix()
        if relative in known_variant_paths:
            continue
        qa = read_json(path.with_suffix(".qa.json"))
        require(qa.get("sha256") == digest(path) and not qa.get("glyph_warnings") and not qa.get("outside_canvas_text")
                and qa.get("readability_pass") is True and float(qa.get("minimum_effective_font_pt", 0)) >= 8,
                f"Additional report variant readability/QA failed: {relative}")
        extras.append({"path": relative, "png_sha256": qa["sha256"]})
    return {"status": "passed", "figure_count": len(rows), "figures": rows, "additional_inserted_variants": extras,
            "note": "Overlap candidates and small type require direct inspection; this is not a visual pass"}


def verify_visual(visual_review, figures_result):
    if visual_review is None:
        return {"status": "pending", "reason": "No recorded direct visual inspection provided"}
    data = read_json(visual_review)
    require(data.get("schema") == "kamp.visual_review.v1", "Unsupported visual-review schema")
    require(data.get("review_method") == "direct_image_inspection" and data.get("reviewer") and data.get("reviewed_at"),
            "Visual review requires direct-inspection method, reviewer and date")
    items = data.get("figures", [])
    require(len(items) == len(FIGURE_IDS) and {x.get("fid") for x in items} == set(FIGURE_IDS), "Visual review must cover all 45 images exactly once")
    hashes = {x["fid"]: x["png_sha256"] for x in figures_result["figures"]}
    variant_hashes = {x["fid"]: x["report_variant"]["sha256"] for x in figures_result["figures"] if x.get("report_variant")}
    for row in items:
        require(row.get("png_sha256") == hashes[row["fid"]], f"Visual review applies to an old image: {row['fid']}")
        require(row.get("verdict") == "pass" and row.get("original_resolution_checked") is True and row.get("unresolved_defects") == [],
                f"Visual review is incomplete or has unresolved defects: {row['fid']}")
        require(bool(row.get("notes")), f"Visual inspection evidence notes missing: {row['fid']}")
        applicability = row.get("pdf_insertion_applicability", "inserted")
        if applicability == "not_inserted":
            require(row.get("pdf_insertion_scale_checked") is False and bool(row.get("non_insertion_reason")),
                    f"Non-insertion needs a reason and truthful unchecked scale status: {row['fid']}")
        else:
            require(applicability == "inserted" and row.get("pdf_insertion_scale_checked") is True,
                    f"Inserted figure was not checked at PDF scale: {row['fid']}")
            if row["fid"] in variant_hashes:
                require(row.get("report_variant_png_sha256") == variant_hashes[row["fid"]],
                        f"Visual review does not cover the actual inserted report variant: {row['fid']}")
    expected_extras = {row["path"]: row["png_sha256"] for row in figures_result.get("additional_inserted_variants", [])}
    extra_rows = data.get("additional_inserted_variants", [])
    require(len(extra_rows) == len(expected_extras) and {r.get("path") for r in extra_rows} == set(expected_extras),
            "Visual review does not cover every additional report variant")
    for row in extra_rows:
        require(row.get("png_sha256") == expected_extras[row["path"]] and row.get("verdict") == "pass"
                and row.get("original_resolution_checked") is True and row.get("pdf_insertion_scale_checked") is True
                and row.get("unresolved_defects") == [] and bool(row.get("notes")),
                f"Additional report variant lacks complete direct inspection: {row['path']}")
    return {"status": "passed", "reviewer": data["reviewer"], "review_sha256": digest(visual_review),
            "reviewed_images": len(items), "note": "Recorded direct inspection checked; actual PDF pages are a separate review"}


def verify_reference(snapshot, reference):
    """Cross-platform baseline comparison against an immutable Windows snapshot."""
    import numpy as np
    import pandas as pd
    current, prior = _load_snapshot(snapshot), _load_snapshot(reference)
    rows = []
    for name in ("oof_long", "test_long"):
        keys = ["model", "data_condition", "datetime"]
        a, b = (ns[name].sort_values(keys).reset_index(drop=True) for ns in (current, prior))
        pd.testing.assert_frame_equal(a[keys], b[keys], check_dtype=False)
        for column in ("y_avg", "y_peak", "y_cls", "fold", "theta", "peak_pred_label"):
            pd.testing.assert_series_equal(a[column], b[column], check_dtype=False, check_exact=True)
        delta = {}
        for column in ("pred_avg", "pred_peak", "prob_raw", "prob_cal", "tau"):
            x, y = a[column].to_numpy(float), b[column].to_numpy(float)
            np.testing.assert_allclose(x, y, atol=1e-6, rtol=1e-6, equal_nan=True)
            valid = np.isfinite(x) & np.isfinite(y)
            delta[column] = float(np.max(np.abs(x[valid] - y[valid]))) if valid.any() else None
        rows.append({"table": name, "rows": len(a), "max_absolute_differences": delta})
    return {"status": "passed", "scope": "baseline OOF/test only; six experiments are independently verified, not cross-platform matched here",
            "reference_metadata_sha256": digest(Path(reference) / "metadata.json"), "atol": 1e-6, "rtol": 1e-6, "comparisons": rows}


def verify_submission(root, snapshot, run_dir, figures, config_path, *, visual_review=None, reference=None):
    gates = {}
    config = read_json(config_path)
    from tools.verify_code_state import verify_code_tests
    from tools.publish_figures import verify_published
    for name, action in (
        ("core", lambda: verify_core(root, snapshot)),
        ("experiments", lambda: verify_experiments(root, snapshot, run_dir, config)),
        ("scientific_quality", lambda: verify_quality(run_dir, config)),
        ("current_code_tests", lambda: verify_code_tests(root, Path(root) / "outputs/verification/current_code_test_manifest.json")),
        ("figure_integrity", lambda: verify_figures(figures, run_dir, snapshot=snapshot, root=root)),
        ("published_figures", lambda: verify_published(root, figures, snapshot)),
    ):
        try:
            gates[name] = action()
        except Exception as exc:
            gates[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if gates["figure_integrity"]["status"] == "passed":
        try:
            gates["visual_review"] = verify_visual(visual_review, gates["figure_integrity"])
        except Exception as exc:
            gates["visual_review"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    else:
        gates["visual_review"] = {"status": "pending", "reason": "Figure integrity must pass first"}
    if reference:
        try:
            gates["baseline_reference"] = verify_reference(snapshot, reference)
        except Exception as exc:
            gates["baseline_reference"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    computation = all(gates[name]["status"] == "passed" for name in
                      ("core", "experiments", "scientific_quality", "current_code_tests", "figure_integrity", "published_figures"))
    failed = any(g["status"] == "failed" for g in gates.values())
    status = "failed" if failed else ("complete" if gates["visual_review"]["status"] == "passed" else "compute_complete_visual_pending")
    result = {"schema": "kamp.submission_verification.v1", "status": status, "scope": "local_code_results_and_figures",
              "computation_complete": computation, "full_ok": status == "complete", "gates": gates,
              "external_validation": {"final_pdf": "not_checked_by_this_tool", "kamp_execution": "not_established_by_local_or_reused_artifacts"}}
    write_json(Path(run_dir) / "verification_summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["strict"], default="strict")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "outputs/analysis_snapshot")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/additional/validated_run")
    parser.add_argument("--figures", type=Path, default=ROOT / "outputs/figures_rerendered")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/config.json")
    parser.add_argument("--visual-review", type=Path)
    parser.add_argument("--reference", type=Path, help="Windows baseline snapshot directory for numeric cross-platform comparison")
    args = parser.parse_args(argv)
    result = verify_submission(ROOT, args.snapshot, args.run_dir, args.figures, args.config,
                               visual_review=args.visual_review, reference=args.reference)
    label = {"complete": "FULL_OK", "compute_complete_visual_pending": "COMPUTE_OK_VISUAL_PENDING", "failed": "VERIFICATION_FAILED"}[result["status"]]
    print(f"{label} scope={result['scope']}")
    return 0 if result["full_ok"] else (3 if result["status"] == "compute_complete_visual_pending" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
