"""Regenerate all 39 figures from a verified snapshot, without importing src modules."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tools.figure_layout import save_figure_with_qa, save_report_variant, refine_figure_layout, exact_rule_summary  # noqa: E402


# Only these static plot expressions are evaluated. Source top-level cells never run.
REGISTRY = {
    "F01": "plot_overall_timeseries(df)",
    "F02": "plot_duplicate_group(df_raw, profile_hashes, profile_dup)",
    "F03": "plot_rowlayout_verification(df_raw, BAD_HOUR_DATES)",
    "F04": "plot_missing_timeline(df)",
    "F05": "plot_hourly_profile(eda_hourly)",
    "F06": "plot_peak_distribution(df, THETA)",
    "F07": "plot_peak_rate_heatmap(df, _THETA_PREVIEW)",
    "F08": "plot_operating_calendar(operating_calendar)",
    "F09": "plot_daily_power_hist(operating_calendar)",
    "F10": "plot_production_vs_power(df, operating_calendar)",
    "F11": "plot_correlation_matrix(df)",
    "F12": "plot_quality_summary(df_raw)",
    "F13": "plot_origin_boundary()",
    "F14": "plot_fold_layout(folds_tbl)",
    "F15": "plot_fold_positive_rate(folds_tbl)",
    "F16": "plot_rf_importance_concentration(rf_importance, rf_feat_names)",
    "F17": "plot_model_mae(regression_tbl)",
    "F18": "plot_pred_vs_actual(FINAL_MODEL_NAME, BEST_BASELINE_NAME)",
    "F19": "plot_pr_curve()",
    "F20": "plot_ablation(ablation_tbl)",
    "F21": "plot_reliability(_calib['y'], _calib['p'], _calib['p_cal'])",
    "F22": "plot_prediction_interval(_unc)",
    "F23": "plot_confusion(_clf_name)",
    "F24": "plot_fold_matrix_heatmap()",
    "F25": "plot_residuals(FINAL_MODEL_NAME)",
    "F26": "plot_optuna_history(hpo_trials)",
    "F27": "plot_permutation(perm_importance)",
    "F28": "plot_shap_summary(shap_values, shap_X)",
    "F29": "plot_condition_mae(condition_tbl, condition_tbl.attrs['overall'])",
    "F30": "plot_fn_fp_by_hour(OOF)",
    "F31": "plot_rule_tree(peak_tree, rule_cols)",
    "F32": "plot_interaction_heatmap(OOF)",
    "F33": "plot_failure_cases(OOF, failure_tbl)",
    "F34": "plot_billing_peaks(tariff_tbl)",
    "F35": "plot_waterfall(levers_tbl)",
    "F36": "plot_net_saving(scenarios_tbl)",
    "F37": "plot_l1_load_curve(PEAK15, p1c)",
    "F38": "plot_sensitivity_heatmap(sensitivity_tbl)",
    "F39": "plot_alert_timeline(protocol_tbl)",
}


def load_plot_namespace(snapshot: dict, root: Path = ROOT) -> tuple[dict, dict]:
    """Compile only named plot definitions and explicit environment style assignments."""
    ns = dict(snapshot)
    ns.update({"np": np, "pd": pd, "plt": plt, "warnings": warnings,
               "LinearSegmentedColormap": LinearSegmentedColormap})
    style_names = {"PALETTE_ADJACENT", "PALETTE_PAIRWISE", "COLOR_HERO", "COLOR_MUTED",
                   "INK", "INK_SOFT", "CMAP_SEQ", "CMAP_DIV"}
    index = {}
    for path in sorted((root / "src").glob("s0*.py")):
        source = path.read_text(encoding="utf-8")
        parsed = ast.parse(source)
        nodes = []
        for node in parsed.body:
            if isinstance(node, ast.FunctionDef) and (node.name.startswith("plot_") or node.name == "setup_korean_font"):
                nodes.append(node)
            elif path.name == "s00_env.py" and isinstance(node, ast.Assign) and all(
                isinstance(target, ast.Name) and target.id in style_names for target in node.targets
            ):
                nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path.name), "exec"), ns)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "save_fig":
                if len(node.args) >= 4 and all(isinstance(x, ast.Constant) for x in node.args[1:4]):
                    fid, title, section = [x.value for x in node.args[1:4]]
                    index[fid] = {"title": title, "section": section, "source": path.name}
    ns["setup_korean_font"]()
    return ns, index


def file_hashes(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def renderer_hashes() -> dict:
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "tools/render_figures.py", ROOT / "tools/figure_layout.py")}


def write_report_manifest(output: Path) -> None:
    directory = output / "report_variants"
    if not directory.exists():
        return
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.qa.json"))]
    parents = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
               for name in ("render_manifest.json", "experiment_render_manifest.json") if (output / name).is_file()}
    manifest = {"schema": "kamp.report_variants.v1", "figures": records,
                "parent_manifest_sha256": parents, "renderer_sha256": renderer_hashes(),
                "visual_status": "requires_direct_image_and_final_pdf_inspection"}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def raw_digest(ns: dict) -> str:
    """Hash numeric contents, excluding pandas' mutable internal lookup caches."""
    keys = ("df_raw", "df", "feat", "OOF", "cv_results", "test_results", "shap_values", "shap_X")
    digest = hashlib.sha256()

    def add(value):
        if isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
            digest.update(pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes())
            if isinstance(value, pd.DataFrame):
                digest.update(str(list(zip(value.columns, map(str, value.dtypes)))).encode("utf-8"))
            else:
                digest.update(str(value.dtype).encode("utf-8"))
        elif isinstance(value, np.ndarray):
            digest.update(str((value.shape, value.dtype)).encode("utf-8"))
            if value.dtype.hasobject:
                add(value.tolist())
            else:
                digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                add(str(key))
                add(value[key])
        elif isinstance(value, (tuple, list)):
            for child in value:
                add(child)
        else:
            digest.update(repr(value).encode("utf-8"))
        digest.update(b"\x00")

    add({k: ns[k] for k in keys if k in ns})
    return digest.hexdigest()


def render_all(snapshot_path: Path, output: Path, *, selected=None) -> dict:
    from tools.analysis_snapshot import load_snapshot

    snapshot_path, output = snapshot_path.resolve(), output.resolve()
    if output == snapshot_path or snapshot_path in output.parents:
        raise ValueError("figure output must be outside the immutable snapshot")
    before = file_hashes(snapshot_path)
    ns, index = load_plot_namespace(load_snapshot(snapshot_path, verify=True))
    # SHAP's beeswarm jitter is visual only; keep it repeatable within an environment.
    np.random.seed(int(ns.get("SEED", 42)))
    before_data = raw_digest(ns)
    output.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for fid in selected or REGISTRY:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fig, source = eval(compile(REGISTRY[fid], "<fixed-plot-registry>", "eval"), ns)
                refine_figure_layout(fig, fid, source)
                path = output / f"{fid}.png"
                qa = save_figure_with_qa(fig, path, fid=fid, dpi=300)
                pd.DataFrame(source).to_csv(output / f"{fid}_src.csv", encoding="utf-8-sig", index=True)
                report_source = exact_rule_summary(ns["peak_tree"], ns["OOF"], ns["rule_cols"], source) if fid == "F31" else source
                variant = save_report_variant(fig, report_source, fid, output / "report_variants", plt=plt)
                if variant:
                    qa["report_variant"] = {"path": f"report_variants/{fid}.png", "sha256": variant["sha256"],
                                            "minimum_effective_font_pt": variant["minimum_effective_font_pt"],
                                            "readability_pass": variant["readability_pass"]}
                plt.close(fig)
            qa["glyph_warnings"] = sorted(set(qa["glyph_warnings"]) | {str(w.message) for w in caught if "Glyph" in str(w.message)})
            qa.update(index.get(fid, {}))
            (output / f"{fid}.qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(qa)
            print(f"{fid}: saved, text overlap candidates={len(qa['overlap_candidates'])}, glyph warnings={len(qa['glyph_warnings'])}", flush=True)
        except Exception as exc:
            failures.append({"fid": fid, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{fid}: FAILED {type(exc).__name__}: {exc}", flush=True)
            plt.close("all")
    unchanged = before == file_hashes(snapshot_path) and before_data == raw_digest(ns)
    manifest = {"schema": "kamp.figure_render.v1", "status": "rendered_pending_visual_review" if not failures and unchanged else "failed",
                "requested": list(selected or REGISTRY), "rendered": len(rows), "failures": failures,
                "snapshot_files_unchanged": before == file_hashes(snapshot_path), "raw_arrays_unchanged": before_data == raw_digest(ns),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "src").glob("s0*.py"))},
                "snapshot_metadata_sha256": hashlib.sha256((snapshot_path / "metadata.json").read_bytes()).hexdigest(),
                "snapshot_fingerprint": ns["metadata"]["fingerprint"], "renderer_sha256": renderer_hashes(),
                "figures": rows, "f19_change": "calendar reference recomputed on common OOF rows; test reference no longer mixed",
                "f38_unit": "10,000 KRW (만원); original currency values unchanged"}
    (output / "render_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report_manifest(output)
    return manifest


def render_experiments(results: Path, output: Path) -> dict:
    """Display the six experiments' recorded results without recalculating them."""
    results, output = results.resolve(), output.resolve()
    ns, _ = load_plot_namespace({})
    output.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []

    def read(name):
        paths = list(results.rglob(name))
        if len(paths) != 1:
            raise ValueError(f"require exactly one {name}, found {len(paths)}")
        return pd.read_csv(paths[0]), paths[0]

    def save(fid, fig, table, input_path, title):
        input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        qa = save_figure_with_qa(fig, output / f"{fid}.png", fid=fid)
        qa.update({"title": title, "source_table": input_path.name, "source_sha256": input_hash})
        table.to_csv(output / f"{fid}_src.csv", encoding="utf-8-sig", index=False)
        variant = save_report_variant(fig, table, fid, output / "report_variants", plt=plt)
        qa["report_variant"] = {"path": f"report_variants/{fid}.png", "sha256": variant["sha256"],
                                "minimum_effective_font_pt": variant["minimum_effective_font_pt"],
                                "readability_pass": variant["readability_pass"]}
        (output / f"{fid}.qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(qa)
        plt.close(fig)

    def rates():
        table, path = read("condition_confusion_rates.csv")
        table = table.copy()
        condition_labels = {"all": "전체", "production_zero": "생산량 0", "production_positive": "생산량 > 0",
                            "production_high": "고생산 (상위 25%)", "hour_08": "08시", "hour_13": "13시",
                            "hour_17_18": "17~18시", "daytime_09_17": "09~17시", "nighttime": "09~17시 외",
                            "shutdown": "휴무일", "operating": "가동일", "first_day_after_shutdown": "휴무 후 첫날"}
        labels = table["condition"].astype(str).map(lambda value: condition_labels.get(value, value))
        fig, axes = plt.subplots(1, 2, figsize=(11, max(4, len(table)*.42)), sharey=True)
        y = np.arange(len(table))
        for ax, key, title in zip(axes, ["miss_rate", "false_alarm_rate"], ["미탐률: FN / 실제 피크 수", "오경보율: FP / 실제 정상 수"]):
            values = table[key].to_numpy(float)
            low, high = table[f"{key}_ci_low"].to_numpy(float), table[f"{key}_ci_high"].to_numpy(float)
            ax.hlines(y, low, high, color=ns["COLOR_MUTED"], lw=2)
            ax.scatter(values, y, color=ns["COLOR_HERO"], s=28, zorder=3)
            for pos in np.flatnonzero(~np.isfinite(values)):
                ax.text(.97, pos, "분모 0 · 비율 미정", ha="right", va="center", fontsize=8)
            ax.set_title(title, fontsize=10)
            ax.set_xlim(-.02, 1.02)
            ax.set_xlabel("비율 · 날짜 블록 재표집 95% 구간", fontsize=9)
        axes[0].set_yticks(y, labels, fontsize=8)
        axes[0].invert_yaxis()
        fig.tight_layout()
        save("E01", fig, table, path, "조건별 미탐률·오경보율")

    def bootstrap():
        table, path = read("bootstrap_block_sensitivity.csv")
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        y = np.arange(len(table))
        ax.hlines(y, table["ci_low"], table["ci_high"], color=ns["COLOR_HERO"], lw=3)
        ax.scatter(table["diff"], y, color=ns["INK"], s=32, zorder=3)
        ax.axvline(0, color=ns["INK_SOFT"], lw=1)
        ax.set_yticks(y, [f"{int(r.block_days)}일 블록 · p={r.p_value_approx:.3f}" for r in table.itertuples()], fontsize=9)
        ax.set_xlabel("RF 대비 MAE 감소량과 95% 구간 (kW)", fontsize=9)
        ax.set_title("블록 길이에 따른 통계적 불확실성 (14일 평가기간)", fontsize=10)
        ax.margins(y=.35)
        fig.tight_layout()
        save("E02", fig, table, path, "블록 부트스트랩 민감도")

    def sensitivity():
        table, path = read("input_sensitivity_metrics.csv")
        table = table.sort_values("mae_delta_c0")
        fig, axes = plt.subplots(1, 2, figsize=(11, max(4, len(table)*.32)), sharey=True)
        y = np.arange(len(table))
        axes[0].barh(y, table["mae_delta_c0"], color=ns["COLOR_HERO"], height=.6)
        axes[0].axvline(0, color=ns["INK_SOFT"], lw=1)
        axes[0].set_xlabel("C0 대비 MAE 변화 (kW)", fontsize=9)
        axes[1].scatter(table["recall"], y, label="Recall", color=ns["COLOR_HERO"], s=28)
        axes[1].scatter(table["f1"], y, label="F1", color=ns["PALETTE_ADJACENT"][1], marker="x", s=28)
        axes[1].set_xlim(-.02, 1.02)
        axes[1].set_xlabel("지표 값 (Recall · F1)", fontsize=9)  # F1 is not a detection rate
        axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -0.09), ncol=2)
        axes[0].set_yticks(y, table["scenario"], fontsize=8)
        axes[0].invert_yaxis()
        fig.suptitle("계획·기상 입력오차 가정 민감도 (실제 계획오차의 관측 결과 아님)", fontsize=11)
        fig.tight_layout()
        save("E03", fig, table, path, "입력오차 민감도")

    def controlled():
        table, path = read("controlled_regime_paired_ci.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for ax, target, title in zip(axes, ["y_avg", "y_peak"], ["평균전력 MAE", "최대전력 MAE"]):
            sub = table[table["target"] == target]
            y = np.arange(len(sub))
            ax.hlines(y, sub["ci_low"], sub["ci_high"], color=ns["COLOR_HERO"], lw=3)
            ax.scatter(sub["MAE_single_minus_regime"], y, color=ns["INK"], s=32, zorder=3)
            ax.axvline(0, color=ns["INK_SOFT"], lw=1)
            ax.set_yticks(y, sub["outer"], fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("단일 - 레짐 MAE (kW) · 양수이면 레짐 우세", fontsize=9)
            ax.margins(y=.35)
        fig.suptitle("동일 회귀설정 통제 비교 · 날짜 블록 95% 구간", fontsize=11)
        fig.tight_layout()
        save("E04", fig, table, path, "레짐 통제 비교")

    def temporal():
        table, path = read("temporal_outer_metrics.csv")
        selected = table[table["selected"].astype(str).str.lower().isin(["true", "1"])].copy()
        if selected.empty:
            raise ValueError("E05 has no selected-model rows")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        x = np.arange(len(selected))
        axes[0].bar(x, selected["MAE"], color=ns["COLOR_HERO"], width=.6)
        axes[0].set_ylabel("평균전력 MAE (kW)", fontsize=9)
        axes[1].plot(x, selected["Recall"], "o-", label="Recall", color=ns["COLOR_HERO"])
        axes[1].plot(x, selected["F1"], "s-", label="F1", color=ns["PALETTE_ADJACENT"][1])
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.set_xticks(x, [f"{r.outer}\n{r.model}" for r in selected.itertuples()], fontsize=8)
        fig.suptitle("후향적 시간순 후보선정 평가 · 각 평가기간 이전에 선택한 모델", fontsize=11)
        fig.tight_layout()
        save("E05", fig, selected, path, "시간순 후보선정 평가")

    def policy():
        table, path = read("policy_cost_sensitivity.csv")
        table = table[np.isclose(table["alpha"], .2) & np.isclose(table["rate_multiplier"], 1)
                      & np.isclose(table["action_cost_krw"], 0) & np.isclose(table["shift_unit_cost_krw"], 0)]
        table = table.drop_duplicates(["outer", "policy"])
        if table.empty:
            raise ValueError("E06 main scenario missing")
        value_column = "assumed_observed_period_net_difference_krw"
        no_complete_cost = table[value_column].isna().all()
        if no_complete_cost:
            value_column = "action_events"
        piv = table.pivot(index="outer", columns="policy", values=value_column)
        if not no_complete_cost:
            piv = piv / 1e4
        order = [p for p in ("none", "fixed_L1", "alert_L1", "alert_L2") if p in piv]
        fig, ax = plt.subplots(figsize=(9, 4))
        piv[order].plot.bar(ax=ax, color=["#b8b7b2", "#2a78d6", "#eb6834", "#1baf7a"][:len(order)], rot=0)
        ax.axhline(0, color=ns["INK_SOFT"], lw=1)
        # A zero result and an undetermined cost both draw no bar; label true zeros.
        for column, container in zip(order, ax.containers):
            for row, bar in zip(piv.index, container):
                value = piv.loc[row, column]
                if pd.notna(value) and value == 0:
                    ax.annotate("0", (bar.get_x() + bar.get_width() / 2, 0), xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7, color=ns["INK_SOFT"])
        # O2..O5 tick labels already identify the evaluation periods; reserve this
        # separate bottom row for the policy legend when partial-coverage labels wrap.
        ax.set_xlabel("")
        ax.set_ylabel("계획된 조치 건수" if no_complete_cost else "가정 내 관측기간 순비용 차이 (만원)", fontsize=9)
        if no_complete_cost:
            ax.set_title("정책별 조치 건수 · 모든 구간에 평가 제외시간이 있어 비용은 미확정\n실제 절감·폐루프 운영 효과를 검증한 결과가 아님", fontsize=10)
        else:
            missing = table[table[value_column].isna()]["outer"].unique()
            if len(missing):
                ax.set_xticklabels([str(v) + ("\n비용 미확정" if v in missing else "") for v in piv.index], rotation=0)
            ax.set_title("정책별 사후 가정 모의 · α=20%, 단가×1, 추가 실행비용=0\n평가 제외 구간은 비용 미확정(눈금 표시) · 숫자 0은 0원 · 실제 절감 실적 아님", fontsize=10)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=4, frameon=False)
        fig.tight_layout()
        save("E06", fig, table, path, "경보 기반 정책 가정 모의")

    for fid, fn in zip([f"E{i:02}" for i in range(1, 7)], [rates, bootstrap, sensitivity, controlled, temporal, policy]):
        try:
            fn()
            print(f"{fid}: saved", flush=True)
        except Exception as exc:
            failures.append({"fid": fid, "error": f"{type(exc).__name__}: {exc}"})
            plt.close("all")
            print(f"{fid}: FAILED {type(exc).__name__}: {exc}", flush=True)
    manifest = {"schema": "kamp.experiment_figures.v1", "status": "rendered_pending_visual_review" if not failures else "failed",
                "figures": rows, "failures": failures, "rendered": len(rows), "renderer_sha256": renderer_hashes()}
    (output / "experiment_render_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report_manifest(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--all", action="store_true", help="render all 39 baseline figures")
    parser.add_argument("--fid", action="append", choices=REGISTRY)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "figures_rerendered")
    parser.add_argument("--experiments", type=Path, help="six-experiment result directory; no training is executed")
    args = parser.parse_args(argv)
    if not args.all and not args.fid and not args.experiments:
        parser.error("provide --all, --fid or --experiments")
    if (args.all or args.fid) and not args.snapshot:
        parser.error("baseline figures require --snapshot")
    manifests = []
    if args.all or args.fid:
        manifests.append(render_all(args.snapshot, args.output, selected=None if args.all else args.fid))
    if args.experiments:
        manifests.append(render_experiments(args.experiments, args.output))
    return 1 if any(m["status"] == "failed" for m in manifests) else 0


if __name__ == "__main__":
    raise SystemExit(main())
