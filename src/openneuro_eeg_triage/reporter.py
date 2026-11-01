"""Outputs: cumulative global tables, the 6-sheet Excel workbook, CSV/JSON summaries and diagnostic figures.

Everything global is rebuilt from *all* ``sub-*/status.json`` files in the output directory. A run on 20 subjects
followed by a run on 10 more therefore produces one workbook over 30 subjects, with ``Overview`` recomputed and
``Subjects`` / ``Recording_Ranking`` re-ranked across all of them.
"""
from __future__ import annotations

import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .config import LABEL_RANK, QC_LOGIC_VERSION, SCORE_WEIGHTS, QCConfig
from .engine import LABEL_TEXT, LAYER_ORDER, LAYER_PREFIX, REASON_CODES, SUBJECT_LABELS, metric, standard_positions
from .utils import Logger, natural_key, now_iso, read_json, write_json_atomic

SHEETS = ["Overview", "Subjects", "Recordings", "Recording_Ranking", "Issues", "Legend"]
USABLE_LABELS = ("USABLE", "USABLE_WITH_FLAGS", "USABLE_NEEDS_INTERPOLATION")


# ================================================================================ loading
def load_all_status(out: Path) -> list[dict[str, Any]]:
    """Every readable ``sub-*/status.json`` in the output directory, in natural subject order."""
    items = []
    for p in sorted(Path(out).glob("sub-*/status.json"), key=lambda q: natural_key(q.parent.name)):
        try:
            st = read_json(p)
        except Exception:
            continue
        if isinstance(st, dict) and st.get("subject"):
            items.append(st)
    return items


def _num(v: Any) -> Any:
    if v in ("", None):
        return None
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in str(v) else f
    except (TypeError, ValueError):
        return v


def _r(v: Any, nd: int = 2) -> Any:
    return round(float(v), nd) if isinstance(v, (int, float, np.floating)) and v is not None and not pd.isna(v) else v


# ================================================================================ quality score
def score_recording(row: dict[str, Any], cfg: QCConfig) -> dict[str, float | None]:
    """0-100 quality score of one recording from seven signal criteria (None where a criterion cannot be
    computed; the remaining weights are renormalised). Used for ranking only, never for the verdict."""
    def clip(x: float) -> float:
        return float(min(1.0, max(0.0, x)))

    nb, mx = row.get("n_bad_channels"), row.get("max_bad_allowed")
    sub: dict[str, float | None] = {k: None for k in SCORE_WEIGHTS}
    if isinstance(nb, (int, float)) and isinstance(mx, (int, float)):
        sub["bad_channels"] = clip(1 - nb / (mx + 1))
        nu = row.get("n_uncorrelated")
        if isinstance(nu, (int, float)):
            sub["uncorrelated_channels"] = clip(1 - nu / (mx + 1))
    nc = row.get("n_critical_bad")
    if isinstance(nc, (int, float)):
        sub["critical_channels"] = clip(1 - nc / max(1, cfg.critical_bad_fail_min))
    corr = row.get("median_spatial_corr")
    if isinstance(corr, (int, float)) and not pd.isna(corr):
        sub["spatial_correlation"] = clip(corr / (3 * cfg.corr_threshold))
    db = row.get("line_db")
    if isinstance(db, (int, float)) and not pd.isna(db):
        sub["line_noise"] = clip(1 - max(db, 0) / cfg.line_high_db)
    fz, dur = row.get("frozen_s"), row.get("duration_s")
    if isinstance(fz, (int, float)) and isinstance(dur, (int, float)) and dur > 0:
        sub["frozen_segments"] = clip(1 - fz / max(1e-9, cfg.global_frozen_fail_fraction * dur))
    if isinstance(dur, (int, float)):
        sub["duration"] = clip(dur / cfg.duration_ok_range_s[0])
    avail = {k: v for k, v in sub.items() if v is not None}
    total_w = sum(SCORE_WEIGHTS[k] for k in avail)
    score = 100.0 * sum(SCORE_WEIGHTS[k] * v for k, v in avail.items()) / total_w if total_w else None
    return {"score": None if score is None else round(score, 1), **{k: (None if v is None else round(v, 3)) for k, v in sub.items()}}


# ================================================================================ tables
def recording_row(rec: dict[str, Any], stale: bool) -> dict[str, Any]:
    L = rec.get("layers", {})
    m = lambda layer, name, d=None: metric(L, layer, name, d)        # noqa: E731
    row = {
        "subject": rec["subject"], "session": rec["session"], "task": rec["task"], "acq": rec["acq"],
        "run": rec["run"], "condition": rec["condition"], "format": rec["format"],
        "overall": rec["overall"], "label": rec["label"], "codes": "; ".join(rec["codes"]),
        **rec["verdicts"],
        "duration_s": _r(m("header", "duration_s")), "n_eeg_channels": rec.get("n_eeg"),
        "sfreq_hz": rec.get("summary", {}).get("sfreq"),
        "max_bad_allowed": m("integrity", "max_bad_allowed"),
        "n_bad_channels": m("integrity", "n_bad_channels"), "bad_channels": ", ".join(m("integrity", "bad_channels", [])),
        "bad_detail": m("integrity", "bad_detail", ""),
        "n_critical_bad": m("integrity", "n_critical_bad"),
        "critical_bad_channels": ", ".join(m("integrity", "critical_bad_channels", [])),
        "n_uncorrelated": m("integrity", "n_uncorrelated"),
        "uncorrelated_channels": ", ".join(m("integrity", "uncorrelated_channels", [])),
        "interpolation_needed": m("integrity", "interpolation_needed", ""),
        "median_spatial_corr": _r(m("integrity", "median_spatial_corr")), "frozen_s": m("integrity", "frozen_s"),
        "typical_amp_uv": _r(m("amplitude", "typical_amplitude_uv")), "median_kurtosis": _r(m("amplitude", "median_kurtosis")),
        "alpha_peak_hz": _r(m("spectral", "alpha_peak_hz")), "alpha_prom_db": _r(m("spectral", "alpha_prominence_db")),
        "alpha_ratio_EC_EO": _r(m("spectral", "alpha_ratio_EC_EO")),
        "alpha_non_responder_possible": m("spectral", "alpha_non_responder_possible", ""),
        "line_hz": m("spectral", "line_hz"), "line_db": _r(m("spectral", "line_db")),
        "line_h2_db": _r(m("spectral", "line_h2_db")), "line_h3_db": _r(m("spectral", "line_h3_db")),
        "notch_suggested": m("spectral", "notch_suggested", ""),
        "line_freq_source": (rec.get("line") or {}).get("source", ""),
        "trigger_events": (rec.get("info") or {}).get("stim_n_events"),
        "load_error": rec.get("error") or "", "rec_id": rec["rec_id"], "config_stale": stale,
    }
    return row


def build_tables(statuses: list[dict[str, Any]], cfg: QCConfig) -> dict[str, pd.DataFrame]:
    """Subjects, Recordings, Recording_Ranking, Issues and the flat full table, over ALL subjects given."""
    sig = cfg.fingerprint()
    rec_rows, issue_rows, full_rows, subj_rows = [], [], [], []
    for st in statuses:
        stale = st.get("config_fingerprint") != sig
        for rec in st.get("records", []):
            row = recording_row(rec, stale)
            parts = score_recording(row, cfg)
            row["quality_score"] = parts["score"]
            for k in SCORE_WEIGHTS:
                row[f"score_{k}"] = parts[k]
            row["usable"] = 1 if rec["label"] in USABLE_LABELS else 0
            rec_rows.append(row)
            base = {"subject": rec["subject"], "session": rec["session"], "task": rec["task"], "acq": rec["acq"], "run": rec["run"]}
            if rec["status"] == "MISSING":
                issue_rows.append({**base, "layer": "load", "code": rec.get("error_code") or "RECORDING_MISSING",
                                   "finding": "recording_missing", "value": rec.get("error"), "limit": "recording must be available", "verdict": "MISSING"})
            elif rec["status"] == "ERROR":
                issue_rows.append({**base, "layer": "load", "code": "FILE_NOT_READABLE", "finding": "file_not_readable",
                                   "value": rec.get("error"), "limit": "file must be readable", "verdict": "FAIL"})
            for layer in LAYER_ORDER:
                for i in rec.get("layers", {}).get(layer, {}).get("issues", []):
                    issue_rows.append({**base, "layer": layer, "code": i["code"], "finding": i["metric"], "value": i["value"],
                                       "limit": i["threshold"], "verdict": i["verdict"]})
            full = {**{k: v for k, v in row.items() if not k.startswith("score_")}}
            for layer in LAYER_ORDER:
                for k, v in rec.get("layers", {}).get(layer, {}).get("metrics", {}).items():
                    if k in ("channel_names", "required_channels_found"):
                        continue
                    full[f"{LAYER_PREFIX[layer]}_{k}"] = "; ".join(map(str, v)) if isinstance(v, list) else v
            full.update({f"info_{k}": v for k, v in (rec.get("info") or {}).items()})
            full["inputs_sha256"] = "; ".join(f"{i['file']}:{i['sha256'][:12]}" for i in rec.get("inputs", []))
            full_rows.append(full)
        s = dict(st.get("summary") or {"subject": st["subject"], "status": st.get("status", "ERROR")})
        s["status"] = st.get("status", s.get("status"))
        s["age"] = _num(s.get("age"))
        s["config_fingerprint"] = st.get("config_fingerprint")
        s["config_stale"] = stale
        s["qc_version"] = st.get("qc_version")
        s["data_deleted"] = (st.get("cleanup") or {}).get("data_deleted")
        s["updated_at"] = st.get("updated_at")
        subj_rows.append(s)

    rec_df = pd.DataFrame(rec_rows)
    if not rec_df.empty:
        rec_df["rec_rank"] = _rank(rec_df)
    rec_df = rec_df.reset_index(drop=True)
    subj_df = pd.DataFrame(subj_rows)
    if not subj_df.empty:
        mean_score = rec_df.groupby("subject")["quality_score"].mean() if not rec_df.empty else pd.Series(dtype=float)
        subj_df["quality_score"] = subj_df["subject"].map(mean_score).round(1)
        usable_group = subj_df["label"].isin(USABLE_LABELS + ("PARTIALLY_USABLE",)).astype(int)
        order = pd.DataFrame({"g": 1 - usable_group, "s": -subj_df["quality_score"].fillna(-1), "k": subj_df["subject"].map(lambda x: tuple(natural_key(x)))})
        order = order.sort_values(["g", "s", "k"], kind="stable")
        ranks = pd.Series(np.arange(1, len(order) + 1), index=order.index)
        subj_df["rank"] = ranks.reindex(subj_df.index).astype(int)
    issues = pd.DataFrame(issue_rows, columns=["subject", "session", "task", "acq", "run", "layer", "code", "finding",
                                               "value", "limit", "verdict"])
    if not issues.empty:
        order_v = {"FAIL": 0, "MISSING": 1, "FLAG": 2, "PASS": 3}
        issues = issues.sort_values("verdict", key=lambda s: s.map(lambda v: order_v.get(v, 9)), kind="stable").reset_index(drop=True)
    return {"subjects": subj_df, "recordings": rec_df, "issues": issues, "full": pd.DataFrame(full_rows)}


def _rank(rec_df: pd.DataFrame) -> pd.Series:
    """Global rank of every recording: usable ones first by score, then the rest; unscored last."""
    key = pd.DataFrame({
        "g": 1 - rec_df["usable"],
        "s": -rec_df["quality_score"].fillna(-1.0),
        "k": [tuple(natural_key(f"{a}_{b}")) for a, b in zip(rec_df["subject"], rec_df["rec_id"])],
    })
    order = key.sort_values(["g", "s", "k"], kind="stable").index
    ranks = pd.Series(np.arange(1, len(order) + 1), index=order)
    return ranks.reindex(rec_df.index)


# ================================================================================ Excel
HEADER_FILL, TITLE_FILL = "1F3864", "D9E1F2"
VERDICT_COLORS = {"PASS": "C6EFCE", "FLAG": "FFEB9C", "FAIL": "FFC7CE", "MISSING": "D9D2E9", "N/A": "EDEDED",
                  "USABLE": "C6EFCE", "USABLE_WITH_FLAGS": "FFEB9C", "USABLE_NEEDS_INTERPOLATION": "FCE4D6",
                  "REJECT": "FFC7CE", "PARTIALLY_USABLE": "FCE4D6", "NO DATA": "EDEDED", "COMPLETED": "C6EFCE",
                  "PARTIAL": "FFEB9C", "NO_DATA": "EDEDED", "ERROR": "FFC7CE"}

SUBJECT_COLS = ["rank", "subject", "status", "verdict", "label", "summary", "quality_score", "sex", "age", "handedness",
                "sessions", "tasks", "n_eeg_channels", "sfreq_hz", "n_recordings", "usable_recordings", "n_PASS",
                "n_FLAG", "n_FAIL", "n_MISSING", "n_USABLE", "n_USABLE_WITH_FLAGS", "n_USABLE_NEEDS_INTERPOLATION",
                "n_REJECT", "bad_channels (recordings flagged)", "critical_channels_bad", "reason_codes",
                "median_alpha_ratio_EC_EO", "median_line_noise_db", "analysis_seconds", "data_deleted", "qc_version",
                "config_fingerprint", "config_stale"]
RECORDING_COLS = ["subject", "session", "task", "acq", "run", "condition", "format", "overall", "label", "codes",
                  "verdict_header", "verdict_integrity", "verdict_amplitude", "verdict_spectral", "verdict_consistency",
                  "duration_s", "n_eeg_channels", "sfreq_hz", "max_bad_allowed", "n_bad_channels", "bad_channels",
                  "n_critical_bad", "critical_bad_channels", "n_uncorrelated", "uncorrelated_channels",
                  "interpolation_needed", "median_spatial_corr", "frozen_s", "typical_amp_uv", "median_kurtosis",
                  "alpha_peak_hz", "alpha_prom_db", "alpha_ratio_EC_EO", "alpha_non_responder_possible", "line_hz",
                  "line_db", "line_h2_db", "line_h3_db", "notch_suggested", "line_freq_source", "trigger_events",
                  "load_error", "quality_score"]
RANKING_COLS = ["rec_rank", "subject", "session", "task", "acq", "run", "quality_score", "overall", "label",
                *[f"score_{k}" for k in SCORE_WEIGHTS], "n_bad_channels", "bad_channels", "codes"]
ISSUE_COLS = ["subject", "session", "task", "acq", "run", "layer", "code", "finding", "value", "limit", "verdict"]


def _cell_value(v: Any) -> Any:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (list, tuple, dict)):
        return str(v)
    if isinstance(v, str) and len(v) > 32000:
        return v[:32000]
    return v


def _write_table(ws, df: pd.DataFrame, cols: list[str], colour_cols: list[str], widths: dict[str, int] | None = None) -> dict[str, str]:
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    cols = [c for c in cols if c in df.columns]
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor=HEADER_FILL)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 32
    for row in df[cols].itertuples(index=False):
        ws.append([_cell_value(v) for v in row])
    letters = {c: get_column_letter(i + 1) for i, c in enumerate(cols)}
    n = len(df)
    for i, c in enumerate(cols, 1):
        w = (widths or {}).get(c)
        if w is None:
            longest = max([len(str(c))] + [len(str(x)) for x in df[c].head(200).tolist()]) if n else len(str(c))
            w = min(max(10, longest + 2), 45)
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2" if "subject" not in cols else f"{get_column_letter(cols.index('subject') + 2)}2"
    if n:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{n + 1}"
        for c in colour_cols:
            if c in letters:
                rng = f"{letters[c]}2:{letters[c]}{n + 1}"
                for val, colour in VERDICT_COLORS.items():
                    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=[f'"{val}"'],
                                                                  fill=PatternFill("solid", bgColor=colour, fgColor=colour)))
    return letters


def _set_default_font(wb) -> None:
    from openpyxl.styles import Font
    try:
        wb._named_styles["Normal"].font = Font(name="Arial", size=10)
        wb._fonts[0] = Font(name="Arial", size=10) if hasattr(wb._fonts, "__setitem__") else wb._fonts[0]
    except Exception:
        pass


def write_excel(path: Path, tables: dict[str, pd.DataFrame], cfg: QCConfig, meta: dict[str, Any]) -> None:
    """The 6-sheet workbook: Overview (live formulas + charts), Subjects, Recordings, Recording_Ranking, Issues, Legend."""
    from openpyxl import Workbook

    subj, rec, issues = tables["subjects"], tables["recordings"], tables["issues"]
    wb = Workbook()
    _set_default_font(wb)
    ws_over = wb.active
    ws_over.title = "Overview"
    ws_s, ws_r = wb.create_sheet("Subjects"), wb.create_sheet("Recordings")
    ws_k, ws_i, ws_l = wb.create_sheet("Recording_Ranking"), wb.create_sheet("Issues"), wb.create_sheet("Legend")

    subj_sorted = subj.sort_values("subject", key=lambda s: s.map(lambda x: tuple(natural_key(x)))) if not subj.empty else subj
    rec_sorted = rec.sort_values(["subject", "rec_id"], key=lambda s: s.map(lambda x: tuple(natural_key(x)))) if not rec.empty else rec
    rank_sorted = rec.sort_values("rec_rank") if not rec.empty else rec
    sl = _write_table(ws_s, subj_sorted, SUBJECT_COLS, ["status", "verdict", "label"])
    rl = _write_table(ws_r, rec_sorted, RECORDING_COLS, ["overall", "label", *[c for c in RECORDING_COLS if c.startswith("verdict_")]])
    kl = _write_table(ws_k, rank_sorted, RANKING_COLS, ["overall", "label"])
    il = _write_table(ws_i, issues, ISSUE_COLS, ["verdict"])
    _write_legend(ws_l, cfg, meta)
    _write_overview(ws_over, subj_sorted, rec_sorted, rank_sorted, issues, sl, rl, kl, il, cfg, meta)
    wb.calculation.fullCalcOnLoad = True
    for ws in wb.worksheets:
        ws.sheet_view.zoomScale = 100
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".qc_summary.", suffix=".xlsx")
        os.close(fd)
        wb.save(tmp)
        os.replace(tmp, path)
    except BaseException:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _write_overview(ws, subj, rec, rank, issues, sl, rl, kl, il, cfg, meta) -> None:
    from openpyxl.chart import BarChart, PieChart, Reference
    from openpyxl.styles import Font, PatternFill

    ns, nr, ni = max(len(subj), 1), max(len(rec), 1), max(len(issues), 1)
    S = lambda col: f"Subjects!${sl[col]}$2:${sl[col]}${ns + 1}" if col in sl else None            # noqa: E731
    R = lambda col: f"Recordings!${rl[col]}$2:${rl[col]}${nr + 1}" if col in rl else None          # noqa: E731
    K = lambda col: f"Recording_Ranking!${kl[col]}$2:${kl[col]}${nr + 1}" if col in kl else None   # noqa: E731
    I = lambda col: f"Issues!${il[col]}$2:${il[col]}${ni + 1}" if col in il else None              # noqa: E731

    bold = Font(name="Arial", size=10, bold=True)
    sect_fill = PatternFill("solid", fgColor=TITLE_FILL)
    ws.column_dimensions["A"].width = 46
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 11
    ws["A1"] = "openneuro-eeg-triage: QC overview"
    ws["A1"].font = Font(name="Arial", size=14, bold=True)
    ws["A2"] = (f"Dataset {meta.get('dataset_id', '')}"
                + (f" ({meta['title']})" if meta.get("title") else "")
                + (f", snapshot {meta['snapshot']}" if meta.get("snapshot") else "")
                + f"  |  QC logic {QC_LOGIC_VERSION}  |  thresholds {cfg.fingerprint()}  |  generated {now_iso()}")
    ws["A3"] = "Every number below is a live formula on the Subjects, Recordings, Recording_Ranking and Issues sheets."
    ws["A3"].font = Font(name="Arial", size=9, italic=True, color="595959")
    row = [5]
    spots: dict[str, tuple[int, int]] = {}

    def section(title: str) -> None:
        r = row[0]
        for c in "ABC":
            ws[f"{c}{r}"].fill = sect_fill
        ws[f"A{r}"] = title
        ws[f"A{r}"].font = bold
        row[0] += 1

    def line(label: str, formula: str, fmt: str | None = None, pct_of: str | None = None) -> int:
        r = row[0]
        ws[f"A{r}"] = label
        ws[f"B{r}"] = formula
        if fmt:
            ws[f"B{r}"].number_format = fmt
        if pct_of:
            ws[f"C{r}"] = f"=IFERROR(B{r}/{pct_of},0)"
            ws[f"C{r}"].number_format = "0.0%"
        row[0] += 1
        return r

    def block(key: str, title: str, items: list[tuple[str, str]], total_ref: str) -> None:
        section(title)
        first = row[0]
        for label, formula in items:
            line(label, formula, pct_of=total_ref)
        spots[key] = (first, row[0] - 1)
        row[0] += 1

    section("Coverage")
    r_sub = line("Subjects in the workbook", f"=COUNTA({S('subject')})")
    for st in ("COMPLETED", "PARTIAL", "NO_DATA", "ERROR"):
        line(f"  status {st}", f'=COUNTIF({S("status")},"{st}")', pct_of=f"$B${r_sub}")
    r_rec = line("Recordings screened", f"=COUNTA({R('subject')})")
    line("Recordings per subject (mean)", f"=IFERROR(B{r_rec}/B{r_sub},0)", fmt="0.0")
    line("Subjects screened with other thresholds (stale)", f'=COUNTIF({S("config_stale")},TRUE)', pct_of=f"$B${r_sub}")
    row[0] += 1

    block("verd", "Recording verdicts", [(v, f'=COUNTIF({R("overall")},"{v}")') for v in ("PASS", "FLAG", "FAIL", "MISSING")], f"$B${r_rec}")
    block("lab", "Recording labels", [(l, f'=COUNTIF({R("label")},"{l}")') for l in (*LABEL_RANK, "MISSING")], f"$B${r_rec}")
    block("slab", "Subject labels", [(l, f'=COUNTIF({S("label")},"{l}")') for l in (*SUBJECT_LABELS, "NO DATA")], f"$B${r_sub}")

    section("Signal metrics (median over recordings)")
    for label, col, fmt in (("Bad channels per recording", "n_bad_channels", "0.0"), ("Median spatial correlation", "median_spatial_corr", "0.00"),
                            ("Typical amplitude (uV)", "typical_amp_uv", "0.0"), ("Mains line noise (dB)", "line_db", "0.0"),
                            ("Alpha ratio Eyes Closed / Eyes Open", "alpha_ratio_EC_EO", "0.00"), ("Duration (s)", "duration_s", "0"),
                            ("Quality score (0-100)", "quality_score", "0.0")):
        line(label, f'=IFERROR(MEDIAN({R(col)}),"n/a")', fmt=fmt)
    line("Recordings with a notch filter suggested", f'=COUNTIF({R("notch_suggested")},TRUE)', pct_of=f"$B${r_rec}")
    line("Recordings that need interpolation", f'=COUNTIF({R("interpolation_needed")},TRUE)', pct_of=f"$B${r_rec}")
    row[0] += 1

    codes = issues["code"].value_counts().head(10).index.tolist() if not issues.empty else []
    if codes:
        block("codes", "Most frequent reason codes", [(c, f'=COUNTIF({I("code")},"{c}")') for c in codes], f"$B${r_rec}")

    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100.01)]
    block("score", "Quality score distribution (recordings)",
          [(f"{lo}-{min(hi, 100):.0f}", f'=COUNTIFS({K("quality_score")},">="&{lo},{K("quality_score")},"<"&{hi})') for lo, hi in bins], f"$B${r_rec}")

    if not subj.empty and subj["age"].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v)).any():
        groups = [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 200)]
        block("age", "Age groups (subjects with a numeric age)",
              [(f"{lo}-{hi - 1}" if hi < 200 else f"{lo}+", f'=COUNTIFS({S("age")},">="&{lo},{S("age")},"<"&{hi})') for lo, hi in groups], f"$B${r_sub}")

    section("In plain words")
    ur = line("Subjects with at least one usable recording", f'=B{r_sub}-COUNTIF({S("label")},"REJECT")-COUNTIF({S("label")},"NO DATA")')
    ws[f"A{row[0]}"] = f'="That is "&TEXT(IFERROR(B{ur}/B{r_sub},0),"0%")&" of the subjects in this workbook."'
    row[0] += 1
    ws[f"A{row[0]}"] = f'="Recordings usable as they are or with flags only: "&TEXT(IFERROR((COUNTIF({R("label")},"USABLE")+COUNTIF({R("label")},"USABLE_WITH_FLAGS"))/B{r_rec},0),"0%")'
    row[0] += 1
    ws[f"A{row[0]}"] = f'="Rejected recordings: "&COUNTIF({R("label")},"REJECT")&" of "&B{r_rec}'

    def bar(key: str, title: str, anchor: str, pie: bool = False) -> None:
        if key not in spots:
            return
        a, b = spots[key]
        ch = PieChart() if pie else BarChart()
        ch.title = title
        ch.height, ch.width = 7.2, 13
        if not pie:
            ch.legend = None
            ch.y_axis.title = "count"
        ch.add_data(Reference(ws, min_col=2, min_row=a, max_row=b), titles_from_data=False)
        ch.set_categories(Reference(ws, min_col=1, min_row=a, max_row=b))
        ws.add_chart(ch, anchor)

    bar("verd", "Recording verdicts", "F5", pie=True)
    bar("lab", "Recording labels", "F21")
    bar("slab", "Subject labels", "F37")
    bar("codes", "Most frequent reason codes", "F53")
    bar("score", "Recording quality score", "F69")
    bar("age", "Subjects by age group", "F85")


def _write_legend(ws, cfg: QCConfig, meta: dict[str, Any]) -> None:
    from openpyxl.styles import Font, PatternFill
    bold = Font(name="Arial", size=10, bold=True)
    fill = PatternFill("solid", fgColor=TITLE_FILL)
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 100
    r = [1]

    def head(t: str) -> None:
        ws[f"A{r[0]}"] = t
        for c in "ABC":
            ws[f"{c}{r[0]}"].fill = fill
        ws[f"A{r[0]}"].font = bold
        r[0] += 1

    def put(a: Any, b: Any = None, c: Any = None) -> None:
        ws[f"A{r[0]}"], ws[f"B{r[0]}"], ws[f"C{r[0]}"] = a, b, c
        r[0] += 1

    head("Scope")
    put("Standard 10-20 / 10-10 / 10-05 montages only", None,
        "Channel labels are normalised (EEG Fz-REF -> Fz, fp1 -> Fp1, T3/T4/T5/T6 -> T7/T8/P7/P8). Recordings with non-standard labels (EGI E1.., BioSemi A1..) are rejected in layer 1 as UNSUPPORTED_MONTAGE, never guessed.")
    put("Triage, not strict QC", None, "Healthy or repairable recordings are kept; only catastrophic ones are rejected (FAIL), each with a reason code.")
    put("Download–Screen–Delete (single-recording peak disk footprint)", None, "Each recording and all its companion files are fetched, screened in memory, its metrics persisted, and the raw files deleted before the next one.")
    r[0] += 1
    head("Subject status")
    for k, v in (("COMPLETED", "every selected recording was screened"), ("PARTIAL", "some recordings could not be downloaded; retried on the next run"),
                 ("NO_DATA", "no recording could be downloaded"), ("ERROR", "unexpected error for this subject")):
        put(k, None, v)
    r[0] += 1
    head("Verdicts (per layer and overall)")
    for k, v in (("PASS", "nothing to note"), ("FLAG", "small problem, the recording is kept"), ("FAIL", "clear problem, the recording is not usable"),
                 ("MISSING", "the recording is not available")):
        put(k, None, v)
    r[0] += 1
    head("Labels")
    for k, v in LABEL_TEXT.items():
        put(k, None, v)
    put("PARTIALLY_USABLE", None, "subject level: some recordings usable, others rejected (keep the usable ones)")
    r[0] += 1
    head("Reason codes")
    put("code", "layer", "meaning (default verdict)")
    for code, (layer, verdict, text) in sorted(REASON_CODES.items(), key=lambda kv: (LAYER_ORDER.index(kv[1][0]) if kv[1][0] in LAYER_ORDER else 9, kv[0])):
        put(code, layer, f"{text} [{verdict}]")
    r[0] += 1
    head("Quality score (0-100, ranking only; never changes a verdict)")
    put("criterion", "weight", "definition")
    defs = {"bad_channels": "1 - bad / (allowed + 1)", "critical_channels": "1 - critical bad / fail limit",
            "uncorrelated_channels": "1 - uncorrelated / (allowed + 1)", "spatial_correlation": "median spatial |r| / (3 x threshold), capped at 1",
            "line_noise": "1 - line dB / HIGH_LINE_NOISE dB", "frozen_segments": "1 - frozen time / fail limit", "duration": "duration / minimum expected duration"}
    for k, w in SCORE_WEIGHTS.items():
        put(k, w, defs[k])
    put("rule", None, "Criteria that cannot be computed are skipped and the remaining weights renormalised. Rejected recordings are always ranked after usable ones.")
    r[0] += 1
    head("Thresholds used by this run (QCConfig)")
    put("key", "value", "adapted rules")
    put("max bad channels", f"floor({cfg.max_bad_channel_ratio:g} x n_eeg)" if cfg.max_bad_channels is None else cfg.max_bad_channels,
        "ratio rounded DOWN, at least 1; or an exact count when --max-bad-channels is given. E.g. 64 ch -> %d, 32 ch -> %d, 19 ch -> %d." % (cfg.max_bad_for(64), cfg.max_bad_for(32), cfg.max_bad_for(19)))
    put("correlation threshold scale", None, f"sqrt(n_eeg / {cfg.corr_reference_n_channels}) between {cfg.corr_min_scale} and 1 (19 ch -> {cfg.corr_scale(19):.2f}, 32 ch -> {cfg.corr_scale(32):.2f}, 64+ ch -> 1.00)")
    put("high-frequency bound", None, f"min({cfg.hf_cap_hz:g} Hz, {cfg.hf_nyquist_fraction:g} x Nyquist) when sfreq <= {cfg.hf_rule_max_sfreq_hz:g} Hz")
    for k, v in cfg.to_dict().items():
        put(k, str(v) if isinstance(v, (list, dict)) else v)
    r[0] += 1
    put("QC logic version", QC_LOGIC_VERSION)
    put("package version", __version__)
    put("threshold fingerprint", cfg.fingerprint())


# ================================================================================ CSV / JSON / orchestration
def write_global_outputs(out: Path, cfg: QCConfig, meta: dict[str, Any], log: Logger | None = None,
                         quiet: bool = False) -> dict[str, Any]:
    """Rebuild ``qc_summary.xlsx``, ``qc_results.csv``, ``qc_subjects.csv`` and ``qc_summary.json`` from every
    subject folder in ``out``. Called after the run (and every few subjects), also after Ctrl+C."""
    out = Path(out)
    statuses = load_all_status(out)
    tables = build_tables(statuses, cfg)
    full, subj, rec = tables["full"], tables["subjects"], tables["recordings"]
    _csv_atomic(out / "qc_results.csv", full)
    _csv_atomic(out / "qc_subjects.csv", subj)
    stale = int(subj["config_stale"].sum()) if not subj.empty else 0
    summary = {
        "tool": "openneuro-eeg-triage", "version": __version__, "qc_logic_version": QC_LOGIC_VERSION,
        "generated_at": now_iso(), "dataset": {k: meta.get(k) for k in ("dataset_id", "title", "snapshot", "source")},
        "backend": meta.get("backend"), "command": meta.get("command"), "config_fingerprint": cfg.fingerprint(), "config": cfg.to_dict(),
        "n_subjects": int(len(subj)), "n_recordings": int(len(rec)),
        "subject_status": dict(Counter(subj["status"])) if not subj.empty else {},
        "subject_labels": dict(Counter(subj["label"])) if not subj.empty else {},
        "recording_verdicts": dict(Counter(rec["overall"])) if not rec.empty else {},
        "recording_labels": dict(Counter(rec["label"])) if not rec.empty else {},
        "subjects_with_other_thresholds": stale,
    }
    write_json_atomic(out / "qc_summary.json", summary)
    xlsx_ok = True
    try:
        write_excel(out / "qc_summary.xlsx", tables, cfg, meta)
    except PermissionError:
        xlsx_ok = False
        if log:
            log.warn("qc_summary.xlsx is open in another program; it will be written again at the next update.")
    if log and not quiet:
        log.info(f"  Summary over {len(subj)} subject(s) / {len(rec)} recording(s) in {out}: "
                 f"qc_summary.xlsx{'' if xlsx_ok else ' (locked)'}, qc_results.csv, qc_subjects.csv, qc_summary.json")
        if stale:
            log.warn(f"{stale} subject(s) in this directory were screened with different thresholds "
                     "(see config_stale in the Subjects sheet). Re-run them with the same settings or use --force.")
    return summary


def _csv_atomic(path: Path, df: pd.DataFrame) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ================================================================================ figures
def _fit_sphere(p: np.ndarray) -> np.ndarray:
    a = np.c_[2 * p, np.ones(len(p))]
    b = (p ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    return sol[:3]


def head_projection(names: list[str]) -> dict[str, tuple[float, float]]:
    """2-D scalp coordinates (azimuthal-equidistant, nose up) of the channels that have a 10-05 template position.
    Template positions only; not individual digitisation."""
    pos = standard_positions()
    pts = {n: pos[n] for n in names if n in pos}
    if len(pts) < 3:
        return {}
    allp = np.array(list(pos.values()))
    c = _fit_sphere(allp)
    out = {}
    v = {n: p - c for n, p in pts.items()}
    theta = {n: math.acos(np.clip(x[2] / np.linalg.norm(x), -1, 1)) for n, x in v.items()}
    tmax = max(theta.values()) or 1.0
    for n, x in v.items():
        phi = math.atan2(x[1], x[0])
        r = theta[n] / tmax * 0.92
        out[n] = (r * math.cos(phi), r * math.sin(phi))
    return out


def _ch_class(r: dict[str, Any]) -> int:
    """0 good, 1 noisy / poor spatial correlation, 2 high line noise or drift, 3 bad (flat / clipped / bad)."""
    if r.get("flat") or r.get("clipped") or r.get("bad"):
        return 3 if (r.get("flat") or r.get("clipped") or r.get("noisy")) else 1
    if r.get("corr_below_thr"):
        return 1
    lt = r.get("line_db")
    if r.get("drift_info") or (lt is not None and lt == lt and lt > 20):
        return 2
    return 0


def plot_subject(out: Path, sub: str, final: list[dict[str, Any]], cfg: QCConfig) -> int:
    """Diagnostic figures of one subject (from the persisted per-recording results; no raw data needed):
    posterior-ROI spectra, channel-quality topomap, amplitude heatmap. Returns the number of figures written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fdir = Path(out) / sub / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    rdir = Path(out) / sub / "recordings"
    items = []
    for r in final:
        if r["status"] != "OK":
            continue
        p = rdir / f"{r['rec_id']}.json"
        if p.exists():
            items.append((r, read_json(p)))
    n_fig = 0
    if not items:
        return 0

    # ---- A: posterior ROI spectra
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = {"closed": "#c0392b", "open": "#2c7fb8", "unknown": "#555555"}
    drawn = False
    for r, cache in items:
        f = rdir / (cache.get("psd_file") or "")
        if not cache.get("psd_file") or not f.exists():
            continue
        z = np.load(f)
        psd = z["psd"].mean(axis=0)
        ax.semilogy(z["freqs"], psd, color=colours.get(r["condition"], "#555"), alpha=0.8, lw=1,
                    label=r["rec_id"].replace(f"{sub}_", ""))
        drawn = True
    if drawn:
        ax.axvspan(*cfg.alpha_band_hz, color="#f1c40f", alpha=0.15)
        hz = next((c.get("line", {}).get("hz") for _, c in items if c.get("line", {}).get("hz")), None)
        if hz:
            ax.axvline(hz, color="#7f8c8d", ls=":", lw=1)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (uV$^2$/Hz), mean of posterior ROI")
        ax.set_title(f"{sub}: posterior ROI spectra (red = eyes closed, blue = eyes open)")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(fdir / f"{sub}_psd_posterior.png", dpi=110)
        n_fig += 1
    plt.close(fig)

    # ---- B: channel quality topomap, one head per recording
    shown = [(r, c) for r, c in items if c.get("ch_table")][:12]
    if shown:
        ncol = min(4, len(shown))
        nrow = math.ceil(len(shown) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.5 * nrow), squeeze=False)
        palette = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (r, c) in zip(axes.ravel(), shown):
            tab = c["ch_table"]
            xy = head_projection([t["channel"] for t in tab])
            ax.set_aspect("equal")
            t = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(t), np.sin(t), color="k", lw=1)
            ax.plot([-0.12, 0, 0.12], [0.99, 1.1, 0.99], color="k", lw=1)
            for row in tab:
                if row["channel"] not in xy:
                    continue
                x, y = xy[row["channel"]]
                k = _ch_class(row)
                ax.scatter([x], [y], s=70, c=palette[k], edgecolors="k", linewidths=0.4, zorder=3)
                if row.get("is_critical") and row.get("bad"):
                    ax.scatter([x], [y], s=190, marker="s", facecolors="none", edgecolors="red", linewidths=2, zorder=4)
                if len(tab) <= 40:
                    ax.text(x, y, row["channel"], fontsize=4.5, ha="center", va="center", zorder=5)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_title(f"{r['rec_id'].replace(sub + '_', '')}\n{r['overall']} / {r['label']}", fontsize=7)
            ax.axis("off")
        from matplotlib.lines import Line2D
        labels = ["good", "noisy / poor correlation", "line noise or drift", "bad (flat / clipped / noisy)"]
        handles = [Line2D([], [], marker="o", ls="", markerfacecolor=c, markeredgecolor="k", markersize=7, label=lb)
                   for c, lb in zip(palette, labels)]
        handles.append(Line2D([], [], marker="s", ls="", markerfacecolor="none", markeredgecolor="red", markeredgewidth=2,
                              markersize=9, label="critical channel bad"))
        fig.suptitle(f"{sub}: channel quality", fontsize=10)
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.09, 1, 0.97))
        fig.savefig(fdir / f"{sub}_channel_quality_topomap.png", dpi=110)
        plt.close(fig)
        n_fig += 1

        # ---- C: amplitude heatmap
        names = [t["channel"] for t in shown[0][1]["ch_table"]]
        mat = np.full((len(names), len(shown)), np.nan)
        bad = np.zeros_like(mat, dtype=bool)
        for j, (_, c) in enumerate(shown):
            by = {t["channel"]: t for t in c["ch_table"]}
            for i, nm in enumerate(names):
                if nm in by:
                    mat[i, j] = by[nm]["window_std_uv"]
                    bad[i, j] = bool(by[nm]["bad"])
        fig, ax = plt.subplots(figsize=(1.2 + 0.7 * len(shown), 1.5 + 0.18 * len(names)))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=np.nanpercentile(mat, 98) if np.isfinite(mat).any() else 1)
        ii, jj = np.where(bad)
        ax.scatter(jj, ii, marker="x", c="red", s=25)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=5)
        ax.set_xticks(range(len(shown)))
        ax.set_xticklabels([r["rec_id"].replace(sub + "_", "") for r, _ in shown], rotation=60, ha="right", fontsize=6)
        fig.colorbar(im, ax=ax, label="typical amplitude (uV)")
        ax.set_title(f"{sub}: typical amplitude per channel (red x = bad)", fontsize=8)
        fig.tight_layout()
        fig.savefig(fdir / f"{sub}_amplitude_heatmap.png", dpi=110)
        plt.close(fig)
        n_fig += 1
    return n_fig
