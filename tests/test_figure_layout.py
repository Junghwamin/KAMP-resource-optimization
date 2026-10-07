"""Plot-only regressions: no source pipeline import or model training."""
import ast
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.figure_layout import audit_figure, save_figure_with_qa, exact_rule_summary
from tools.render_figures import load_plot_namespace, REGISTRY, render_experiments, file_hashes


def test_plot_loader_does_not_import_pipeline():
    before = {name for name in sys.modules if name.startswith(tuple(f"s{i:02}" for i in range(11)))}
    ns, index = load_plot_namespace({})
    after = {name for name in sys.modules if name.startswith(tuple(f"s{i:02}" for i in range(11)))}
    assert before == after
    assert set(REGISTRY) <= set(index)
    assert len(REGISTRY) == 39
    for expression in REGISTRY.values():
        call = ast.parse(expression, mode="eval").body
        assert call.func.id in ns


def test_save_does_not_restore_heatmap_grid_or_tick_size(tmp_path):
    fig, ax = plt.subplots()
    ax.imshow([[1, 2], [3, 4]])
    ax.grid(False)
    ax.tick_params(labelsize=7)
    qa = save_figure_with_qa(fig, tmp_path / "figure.png", fid="fixture")
    assert not any(line.get_visible() for line in ax.get_xgridlines()+ax.get_ygridlines())
    assert all(t.get_fontsize() == 7 for t in ax.get_xticklabels())
    assert qa["dpi"] == 300
    assert not qa["glyph_warnings"]
    plt.close(fig)


def test_qa_detects_overlapping_annotations():
    fig, ax = plt.subplots()
    ax.text(.5, .5, "abc", fontsize=12)
    ax.text(.5, .5, "def", fontsize=12)
    qa = audit_figure(fig, "test")
    assert any({p["a"], p["b"]} == {"abc", "def"} for p in qa["overlap_candidates"])
    plt.close(fig)


def test_f19_reference_uses_identical_oof_rows_and_truth():
    idx = pd.date_range("2021-02-01 08:00", periods=24, freq="h")
    truth = np.array([1, 1, 0, 0]*6)
    frame = pd.DataFrame({"y_cls": truth, "prob": np.linspace(0, 1, 24), "pred_peak": np.arange(24)}, index=idx)
    ns, _ = load_plot_namespace({"BEST_MAE_MODEL": "LightGBM", "cv_results": {
        "피크 직접분류": {"oof": frame}, "LightGBM": {"oof": frame.copy()}}})
    fig, source = ns["plot_pr_curve"]()
    ref = source[source["모델"] == "달력규칙"].iloc[0]
    pred = (idx.dayofweek < 5) & (idx.hour >= 8) & (idx.hour <= 18)
    assert ref["TP"] == np.sum(pred & (truth == 1))
    assert ref["FP"] == np.sum(pred & (truth == 0))
    assert set(source["n"]) == {24}
    assert set(source["평가구간"]) == {"공통 OOF"}
    plt.close(fig)


def test_cost_annotation_precision_preserves_small_difference():
    ns, _ = load_plot_namespace({})
    table = pd.DataFrame({"시나리오": ["L1c", "S2"], "순절감액(원)": [915755, 924176]})
    fig, source = ns["plot_net_saving"](table)
    labels = [text.get_text() for text in fig.axes[0].texts]
    assert "91.58" in labels and "92.42" in labels
    pd.testing.assert_frame_equal(source.reset_index(drop=True), table)
    plt.close(fig)


def test_rule_tree_accepts_safe_json_and_shows_raw_counts():
    oof = pd.DataFrame({"시간": [1, 2, 8, 9], "y_cls": [0, 0, 1, 0]})
    tree = {"nodes": [{"id": 0, "left": 1, "right": 2, "feature": 0, "threshold": 5},
                       {"id": 1, "left": -1, "right": -1, "feature": -2, "threshold": -2},
                       {"id": 2, "left": -1, "right": -1, "feature": -2, "threshold": -2}]}
    ns, _ = load_plot_namespace({"OOF": oof, "FEATURE_LABELS": {}, "rules_tbl": pd.DataFrame()})
    fig, _ = ns["plot_rule_tree"](tree, ["시간"])
    assert any("n=4 · 피크 1" in text.get_text() for text in fig.axes[0].texts)
    plt.close(fig)


def test_rule_summary_uses_raw_counts_without_double_rounding():
    oof = pd.DataFrame({"x": [0]*21 + [1]*87, "y_cls": [1]*4 + [0]*17 + [1]*5 + [0]*82})
    tree = {"nodes": [{"id": 0, "left": 1, "right": 2, "feature": 0, "threshold": .5},
                       {"id": 1, "left": -1, "right": -1, "actual_n": 21, "actual_peak": 4},
                       {"id": 2, "left": -1, "right": -1, "actual_n": 87, "actual_peak": 5}]}
    rounded = pd.DataFrame({"규칙": ["R2", "R3"], "피크 확률": [.1905, .0575], "n": [21, 87]})
    result = exact_rule_summary(tree, oof, ["x"], rounded)
    assert result["실제 피크 수"].tolist() == [4, 5]
    assert [f"{v:.1%}" for v in result["원시 피크 비율"]] == ["19.0%", "5.7%"]
    assert rounded["피크 확률"].tolist() == [.1905, .0575]


def test_all_experiment_csv_contracts_render_without_modifying_inputs(tmp_path):
    inputs, output = tmp_path / "results", tmp_path / "figures"
    inputs.mkdir()
    fixtures = {
        "condition_confusion_rates.csv": [{"condition": "전체", "miss_rate": .2, "miss_rate_ci_low": .1,
            "miss_rate_ci_high": .3, "false_alarm_rate": .1, "false_alarm_rate_ci_low": .05, "false_alarm_rate_ci_high": .15}],
        "bootstrap_block_sensitivity.csv": [{"block_days": 1, "diff": 1, "ci_low": -.2, "ci_high": 2, "p_value_approx": .08}],
        "input_sensitivity_metrics.csv": [{"scenario": "C0", "mae_delta_c0": 0, "recall": .8, "f1": .4}],
        "controlled_regime_paired_ci.csv": [{"outer": "O2", "target": t, "MAE_single_minus_regime": 1,
            "ci_low": -1, "ci_high": 3} for t in ["y_avg", "y_peak"]],
        "temporal_outer_metrics.csv": [{"outer": "O2", "model": "lgb", "selected": True, "MAE": 6, "Recall": .8, "F1": .4}],
        "policy_cost_sensitivity.csv": [{"outer": outer, "policy": p, "alpha": .2, "rate_multiplier": 1,
            "action_cost_krw": 0, "shift_unit_cost_krw": 0,
            "assumed_observed_period_net_difference_krw": np.nan if outer == "O2" else v}
            for outer in ["O2", "O3"]
            for p, v in [("none", 0), ("fixed_L1", 100), ("alert_L1", 200), ("alert_L2", -50)]],
    }
    for name, records in fixtures.items():
        pd.DataFrame(records).to_csv(inputs / name, index=False, encoding="utf-8-sig")
    before = file_hashes(inputs)
    result = render_experiments(inputs, output)
    assert not result["failures"]
    assert result["rendered"] == 6
    assert before == file_hashes(inputs)
    assert not any(row["glyph_warnings"] for row in result["figures"])
    assert not next(row for row in result["figures"] if row["fid"] == "E06")["overlap_candidates"]
    assert all(row["report_variant"]["minimum_effective_font_pt"] >= 8 for row in result["figures"])
    assert (output / "report_variants" / "manifest.json").is_file()
    # Quality-excluded hours must not become a zero-cost bar or a realized saving.
    table = pd.DataFrame(fixtures["policy_cost_sensitivity.csv"])
    table["assumed_observed_period_net_difference_krw"] = np.nan
    table["action_events"] = [0, 10, 4, 2] * 2
    table.to_csv(inputs / "policy_cost_sensitivity.csv", index=False)
    partial = render_experiments(inputs, output)
    assert not partial["failures"]
