"""Plot-only helpers: no training imports and no mutation of source data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from matplotlib.text import Text
from PIL import Image


# Actual aspect-fit boxes in the canonical PDF, in points (72 points = 1 inch).
REPORT_BOXES = {
    "F05": (419.774, 169.375), "F07": (424.931, 266.349), "F02": (435.246, 148.398),
    "F14": (444.721, 135.572), "F18": (424.931, 119.389), "F24": (424.931, 223.915),
    "F19": (376.597, 290.682), "F20": (424.931, 125.623), "F27": (424.931, 243.81),
    "F29": (424.931, 181.12), "F30": (424.931, 146.60), "F33": (424.931, 108.00),
    "F31": (424.931, 175.01), "F35": (424.931, 158.47), "F21": (332.82, 305.07),
    "F31_landscape": (781, 384),
    **{f"E{i:02}": (511, h) for i, h in enumerate([270, 310, 195, 290, 305, 215], 1)},
}


def audit_figure(fig, fid: str) -> dict:
    """Report text-box candidates; visual inspection adjudicates intentional overlap."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    seen = set()
    outside_ticks = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if not ax.axison or not axis.get_visible():
                outside_ticks.update((id(axis.label), id(axis.offsetText)))
            lo, hi = sorted(axis.get_view_interval())
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                if not ax.axison or not axis.get_visible() or not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    outside_ticks.update((id(tick.label1), id(tick.label2)))
    for obj in fig.findobj(Text):
        if id(obj) in seen or id(obj) in outside_ticks or not obj.get_visible() or not obj.get_text().strip():
            continue
        seen.add(id(obj))
        box = obj.get_window_extent(renderer)
        if not np.isfinite(box.extents).all() or box.width <= 0 or box.height <= 0:
            continue
        texts.append((obj, box))
    overlaps = []
    for i, (a, ba) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            # A small anti-aliasing fringe is not a collision.
            left, bottom = max(ba.x0, bb.x0), max(ba.y0, bb.y0)
            right, top = min(ba.x1, bb.x1), min(ba.y1, bb.y1)
            if right - left > 2 and top - bottom > 2:
                overlaps.append({"a": a.get_text(), "b": b.get_text(),
                                 "overlap_px": [round(right-left, 2), round(top-bottom, 2)]})
    outside = [{"text": text.get_text(), "bbox_px": list(box.extents)} for text, box in texts
               if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1]
    return {"fid": fid, "text_count": len(texts), "overlap_candidates": overlaps,
            "outside_canvas_text": outside,
            "minimum_font_pt": min((x.get_fontsize() for x, _ in texts), default=None),
            "visual_status": "pending", "qa_scope": "candidate detection; visual review required",
            "intentional_data_overlap": "line intersections, error bars and SHAP dots are not text defects"}


def save_figure_with_qa(fig, path, *, fid: str, dpi: int = 300, tight: bool = True) -> dict:
    """Save without re-enabling grid or overriding ticks/fonts after layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qa = audit_figure(fig, fid)
        fig.savefig(path, dpi=dpi, bbox_inches="tight" if tight else None, pad_inches=.10 if tight else 0, facecolor="white")
    with Image.open(path) as png:
        pixel_size = list(png.size)
    qa.update({"dpi": dpi, "png": path.name,
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "pixel_size": pixel_size, "saved_size_pt": [v / dpi * 72 for v in pixel_size],
               "glyph_warnings": sorted({str(w.message) for w in caught if "Glyph" in str(w.message)}),
               "other_warnings": sorted({str(w.message) for w in caught if "Glyph" not in str(w.message)})})
    path.with_suffix(".qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    return qa


def refine_figure_layout(fig, fid, source):
    """Final per-figure geometry corrections; scientific source functions stay frozen."""
    ax = fig.axes[0]
    if fid in ("F12", "F16"):
        low, high = ax.get_xlim()
        ax.set_xlim(low, high + (high-low) * .12)
    if fid in ("F15", "F17", "F35"):
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + (high-low) * .16)
    if fid == "F35":
        labels = []
        for value in source["레버"]:
            text = str(value)
            if " (" in text:
                text = text.replace(" (", "\n(")
            elif " 시간대 " in text:
                text = text.replace(" 시간대 ", "\n시간대 ")
            elif " 생산량 " in text:
                text = text.replace(" 생산량 ", "\n생산량 ")
            labels.append(text)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=0, ha="center")
    if fid == "F37":
        for text in ax.texts:
            if text.get_text().startswith("08시 "):
                text.set_position((.10, .80))
    if fid == "F08":
        # Snapshot round-trips store dates as timestamps; show the dates only.
        for text in ax.texts:
            text.set_text(text.get_text().replace(" 00:00:00", ""))
    if fid == "F24":
        _contrast_cell_labels(ax)
    if fid == "F25":
        ax.set_ylabel(ax.get_ylabel().rstrip(")") + ", kW)" if ax.get_ylabel().endswith(")") else ax.get_ylabel() + " (kW)")
    if fid == "F27":
        # Fold-to-fold error bars were too light to read on grey bars.
        for collection in ax.collections:
            collection.set_color("#52514e")
            collection.set_linewidth(1.0)
    if fid == "F06":
        # Threshold labels sat exactly on their vertical lines; keep a 2-pt gap.
        for text in ax.texts:
            if text.get_horizontalalignment() == "right":
                _nudge_text(text, -2 * fig.dpi / 72)
    if fid == "F26":
        # Explain the grey per-trial values and the blue running best, with units.
        for panel in fig.axes:
            if len(panel.lines) >= 2:
                panel.legend(panel.lines[:2], ["trial별 값", "누적 최적값"], fontsize=8, frameon=False, loc="best")
            title = panel.get_title()
            if "MAE" in title:
                panel.set_ylabel("MAE (kW)", fontsize=9)
            elif "PR-AUC" in title:
                panel.set_ylabel("PR-AUC", fontsize=9)
    if fid == "F29":
        # Value labels that started left of the overall-mean line were cut by it;
        # start them just right of the line (data units hold at report scale too).
        verticals = [line.get_xdata()[0] for line in ax.lines if len(set(np.atleast_1d(line.get_xdata()))) == 1]
        if verticals:
            overall = float(verticals[0])
            for text in ax.texts:
                x, y = text.get_position()
                if x < overall:
                    text.set_position((overall + overall * .03, y))
    if fid == "F32":
        fig.axes[-1].set_ylabel("평균 peak15 (kW)", fontsize=8)  # colorbar unit
    if fid == "F38":
        ax.set_xlabel("기본요금 단가 배수", fontsize=9)
    if fid in ("F12", "F15", "F16", "F17", "F35", "F37", "F25", "F26", "F32", "F38"):
        fig.canvas.draw()
        fig.tight_layout()


def _nudge_text(text, dx_px):
    """Move a Text or Annotation horizontally by dx display pixels in its own
    coordinate system (an Annotation ignores a composed transform)."""
    from matplotlib.text import Annotation
    if isinstance(text, Annotation):
        x, y = text.xyann
        coords = text.anncoords
        if coords == "offset points":
            text.xyann = (x + dx_px * 72 / text.figure.dpi, y)
        elif coords == "offset pixels":
            text.xyann = (x + dx_px, y)
        elif coords == "data":
            trans = text.axes.transData
            px, py = trans.transform((x, y))
            text.xyann = (trans.inverted().transform((px + dx_px, py))[0], y)
        else:
            raise ValueError(f"Unsupported annotation coordinates for nudging: {coords}")
    else:
        trans = text.get_transform()
        px, py = trans.transform(text.get_position())
        text.set_position(trans.inverted().transform((px + dx_px, py)))


def _keep_texts_inside(ax, pad_pt=2.5):
    """Shift annotations horizontally so their ink stays inside the axes frame."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    pad = pad_pt * fig.dpi / 72
    for text in ax.texts:
        box = text.get_window_extent(renderer)
        shift = 0.
        if box.x1 > frame.x1 - pad:
            shift = (frame.x1 - pad) - box.x1
        elif box.x0 < frame.x0 + pad:
            shift = (frame.x0 + pad) - box.x0
        if shift:
            _nudge_text(text, shift)


def _contrast_cell_labels(ax):
    """Pick black or white cell text from the cell's own colour (WCAG contrast)."""
    mappables = list(ax.images) + [c for c in ax.collections if c.get_array() is not None]
    if not mappables:
        return
    mappable = mappables[0]

    def luminance(rgb):
        channel = [c / 12.92 if c <= .03928 else ((c + .055) / 1.055) ** 2.4 for c in rgb[:3]]
        return .2126 * channel[0] + .7152 * channel[1] + .0722 * channel[2]

    for text in ax.texts:
        try:
            value = float(text.get_text().replace(",", ""))
        except ValueError:
            continue
        background = luminance(mappable.to_rgba(value))
        white, black = 1.05 / (background + .05), (background + .05) / .05
        text.set_color("white" if white >= black else "black")


def exact_rule_summary(tree, oof, cols, rules):
    """Recover the selected rule rates from raw OOF leaf counts, never rounded rates."""
    nodes = {int(node["id"]): node for node in tree["nodes"]}
    leaves = []

    def walk(node_id, mask):
        node = nodes[node_id]
        n = int(mask.sum())
        peaks = int(oof.loc[mask, "y_cls"].sum())
        if "actual_n" in node and (n != node["actual_n"] or peaks != node["actual_peak"]):
            raise ValueError("F31 snapshot tree counts disagree with raw OOF rows")
        if int(node["left"]) < 0:
            if n:
                leaves.append((n, peaks, peaks / n))
            return
        feature = node["feature"]
        column = cols[int(feature)] if isinstance(feature, (int, np.integer)) else feature
        left = oof[column] <= float(node["threshold"])
        walk(int(node["left"]), mask & left)
        walk(int(node["right"]), mask & ~left)

    walk(0, np.ones(len(oof), dtype=bool))
    result = rules.copy()
    numerators, rates = [], []
    for _, row in rules.iterrows():
        matches = {(peaks, rate) for n, peaks, rate in leaves
                   if n == int(row["n"]) and np.isclose(round(rate, 4), row["피크 확률"], atol=1e-12, rtol=0)}
        if len(matches) != 1:
            raise ValueError("F31 selected rule cannot be matched uniquely to raw leaf counts")
        peaks, rate = matches.pop()
        numerators.append(peaks)
        rates.append(rate)
    result["실제 피크 수"] = numerators
    result["원시 피크 비율"] = rates
    return result


def save_report_variant(fig, source, fid, output, *, plt):
    """Re-layout at the actual PDF box with >=8.6-point text, preserving plotted data.

    Summary variants are explicit: F31 shows the three recorded rule rates, with a
    separate full tree; E03 shows every scenario's MAE change (other metrics stay
    in the appendix table). The full canonical figures remain available.
    """
    from matplotlib.ticker import MaxNLocator

    if fid not in REPORT_BOXES:
        return None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    width, height = REPORT_BOXES[fid]
    notes = []
    if fid == "F31":
        landscape = save_report_variant(fig, source, "F31_landscape", output, plt=plt)
        fig, ax = plt.subplots()
        y = np.arange(len(source))
        rates = source["원시 피크 비율"]
        ax.barh(y, rates * 100, color="#2a78d6", height=.55)
        ax.set_yticks(y, source["규칙"])
        ax.invert_yaxis()
        for pos, (_, row) in enumerate(source.iterrows()):
            ax.text(row["원시 피크 비율"] * 100 + 1, pos,
                    f"{row['원시 피크 비율']:.1%} · n={int(row['n']):,}", va="center")
        ax.set_xlim(0, min(120, max(rates) * 100 + 32))
        ax.set_xlabel("실제 피크 비율 (%)")  # the body caption cites the full-tree appendix
        notes.append("본문은 R1~R3 실제 비율·n 요약; 전체 트리는 F31_landscape.png")
        notes.append("원시 OOF의 실제 피크 수 / 실제 n으로 표시하여 4자리 비율의 이중 반올림 방지")
        notes.append(f"landscape_min_effective_font_pt={landscape['minimum_effective_font_pt']:.3f}")
    elif fid == "E03":
        fig, ax = plt.subplots()
        x = np.arange(len(source))
        ax.bar(x, source["mae_delta_c0"], color="#2a78d6", width=.65)
        ax.axhline(0, color="#52514e", lw=.8)
        ax.set_xticks(x, source["scenario"], rotation=45, ha="right")
        ax.set_ylabel("MAE 변화 (kW)")
        ax.set_title("가정한 입력오차: C0 대비 MAE 변화 (17개 시나리오)")
        notes.append("모든 시나리오 MAE 변화 요약; Recall/F1 등은 바로 아래 수치표")
    fig.set_size_inches(width / 72, height / 72, forward=True)
    fig.set_layout_engine(None)
    for ax in fig.axes:
        if fid not in ("F02", "F20", "F33", "F31_landscape", "E01", "E03", "E04", "E06"):
            ax.set_title("")
        if fid == "F02":
            ax.set_title("15일 전력 프로파일 일치" if ax is fig.axes[0] else "동일 날짜의 기온 차이")
            if ax is fig.axes[1]:
                low, high = ax.get_ylim()
                ax.set_ylim(low-2, high+4)
        if fid == "F20":
            # Keep the direction cue: the same orange means opposite things per panel.
            ax.set_title("MAE 변화 (kW) · 클수록 중요" if ax is fig.axes[0] else "Recall 변화 · 음수일수록 중요")
            ax.set_xlabel("")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        if fid == "F35":
            ax.set_xticks(range(len(source)))
            ax.set_xticklabels([str(value).split()[0] for value in source["레버"]], rotation=0, ha="center")
            ax.set_ylim(min(0, source["Δ최대수요(kW)"].min()), max(1, source["Δ최대수요(kW)"].max()) * 1.2)
            notes.append("레버 이름은 본문 정의와 대응하는 코드로 축약")
        if fid == "F14":
            import re
            import matplotlib.dates as mdates
            for text in ax.texts:
                match = re.search(r"양성 (\d+)건", text.get_text())
                if match:
                    text.set_text(("제외" if text.get_text().startswith("폐기") else "사용") + f" · 피크 {match.group(1)}건")
                text.set_transform(ax.get_yaxis_transform())
                text.set_position((1.02, text.get_position()[1]))
            ax.set_xlim(min(p.get_x() for p in ax.patches), max(p.get_x()+p.get_width() for p in ax.patches))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m월"))
            notes.append("본문 fold 도식은 상태·피크건수만 병기; 휴무 비중은 전체 F14 원그림에 보존")
        if fid == "F18":
            import matplotlib.dates as mdates
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_ylabel("평균전력(kW)")
            leg = ax.get_legend()
            if leg:
                labels = [t.get_text().replace(" 예측", "").replace("2단계 레짐(3분류)", "최종 레짐")
                          .replace("Random Forest (보정)", "보정 RF") for t in leg.get_texts()]
                for text, label in zip(leg.get_texts(), labels):
                    text.set_text(label)
                handles = ax.get_legend_handles_labels()[0]
                leg.remove()
                ax.legend(handles, labels, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0),
                          frameon=False, fontsize=8.6, handlelength=1.5, columnspacing=1.0)
        if fid == "F29":
            leg = ax.get_legend()
            if leg:
                reference = leg.get_texts()[0].get_text()
                leg.remove()
                ax.set_xlabel(f"MAE (kW) · 기준선: {reference}")
        if fid == "F33":
            ax.set_title(ax.get_title().replace("사례 ", "").replace(" — ", ": "))
            ax.set_xlabel("시각(시)" if ax is fig.axes[-1] else "")
            ax.set_ylabel("평균전력 (kW)" if ax is fig.axes[0] else "")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        if fid == "F20" and ax is fig.axes[0]:
            ax.set_yticks(range(5), ["과거전력", "생산·인원", "공정상태", "기상", "레짐"])
    if fid == "F33":
        # A figure-level legend collided with the tick labels in the short box;
        # the empty upper area of the last panel holds it without overlap.
        holder = next((panel for panel in fig.axes if panel.get_legend()), None)
        if holder is not None:
            handles, labels = holder.get_legend_handles_labels()
            holder.get_legend().remove()
            fig.axes[-1].legend(handles, labels, loc="upper right", fontsize=8.6, frameon=False)
    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)
    for text in fig.findobj(Text):
        if text.get_visible():
            text.set_fontsize(8.6 if fid != "F31_landscape" else 9.2)
    if fid == "F31_landscape":
        for text in fig.axes[0].texts:
            if "n=" in text.get_text():
                text.set_fontsize(10)
        fig.axes[0].set_title("피크 규칙 트리 · 가중치 없는 실제 표본수와 피크 수", fontsize=11)
    fig.canvas.draw()
    fig.tight_layout(pad=.45, w_pad=.65, h_pad=.65)
    fig.canvas.draw()
    fig.tight_layout(pad=.45, w_pad=.65, h_pad=.65)
    if fid == "F02":
        # The 19.2 °C label over the last bar touched the right spine at 8.6 pt.
        _keep_texts_inside(fig.axes[1])
    # Plot legends sometimes inherit an external anchor too distant for the small box.
    if fid == "F19":
        fig.axes[0].get_legend().set_bbox_to_anchor((.5, -.15))
        fig.tight_layout(pad=.45)
    path = output / f"{fid}.png"
    qa = save_figure_with_qa(fig, path, fid=fid, tight=False)
    scale = min(width / qa["saved_size_pt"][0], height / qa["saved_size_pt"][1])
    qa.update({"variant": "report_physical_box", "target_pdf_box_pt": [width, height],
               "minimum_effective_font_pt": qa["minimum_font_pt"] * scale,
               "aspect_fit_scale": scale, "layout_notes": notes,
               "readability_pass": qa["minimum_font_pt"] * scale >= 8 and not qa["outside_canvas_text"]})
    if fid == "F31":
        source_path = output / "F31_src.csv"
        source.to_csv(source_path, index=False, encoding="utf-8-sig")
        qa.update(source_table=source_path.name, source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
    path.with_suffix(".qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    if fid in ("F31", "E03"):
        plt.close(fig)
    return qa


def draw_rule_tree(tree, cols, oof, labels, plt):
    """Draw compact nodes with observed counts; accepts estimator or safe snapshot JSON."""
    if isinstance(tree, dict):
        nodes = tree["nodes"]
    else:
        t = tree.tree_
        nodes = [dict(id=i, left=int(t.children_left[i]), right=int(t.children_right[i]),
                      feature=int(t.feature[i]), threshold=float(t.threshold[i]),
                      n_node_samples=int(t.n_node_samples[i])) for i in range(t.node_count)]
    by_id = {int(n["id"]): n for n in nodes}
    leaf_order = []
    xy, counts = {}, {}

    def visit(node_id, mask, depth):
        node = by_id[node_id]
        counts[node_id] = (int(mask.sum()), int(oof.loc[mask, "y_cls"].sum()))
        left, right = int(node["left"]), int(node["right"])
        if left < 0:
            x = len(leaf_order)
            leaf_order.append(node_id)
        else:
            feature = node["feature"]
            column = cols[int(feature)] if isinstance(feature, (int, np.integer)) else feature
            m = oof[column] <= float(node["threshold"])
            x = (visit(left, mask & m, depth + 1) + visit(right, mask & ~m, depth + 1)) / 2
        xy[node_id] = (x, -depth)
        return x

    visit(0, np.ones(len(oof), dtype=bool), 0)
    fig, ax = plt.subplots(figsize=(12, 5.6))
    for node_id, node in by_id.items():
        x, y = xy[node_id]
        for key, answer in (("left", "예"), ("right", "아니오")):
            child = int(node[key])
            if child >= 0:
                cx, cy = xy[child]
                ax.plot([x, cx], [y-.16, cy+.16], color="#a3a3a3", lw=.8, zorder=1)
                ax.text((x+cx)/2, (y+cy)/2, answer, ha="center", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1))
        n, peak = counts[node_id]
        if int(node["left"]) >= 0:
            feature = node["feature"]
            column = cols[int(feature)] if isinstance(feature, (int, np.integer)) else feature
            heading = f"{labels.get(column, column)}\n≤ {node['threshold']:.1f}"
        else:
            heading = "말단 조건"
        text = f"{heading}\nn={n:,} · 피크 {peak:,}"
        ax.text(x, y, text, ha="center", va="center", fontsize=9, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=.45", facecolor="#eaf2fc", edgecolor="#7a9cbf"), zorder=3)
    ax.set_xlim(-.65, max(1, len(leaf_order)-1)+.65)
    ax.set_ylim(min(y for _, y in xy.values())-.5, .55)
    ax.set_title("피크 발생조건 규칙트리 (깊이 3, OOF)\n노드 수치는 가중치 없는 실제 표본수와 실제 피크 수", fontsize=11, pad=15)
    ax.axis("off")
    fig.tight_layout()
    return fig
