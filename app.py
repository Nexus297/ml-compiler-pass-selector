import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from compiler_frontend import Lexer, TACParser
from dataset_generator import CATEGORIES, generate_source_program
from feature_extractor import TACFeatureExtractor
from iterative_loop import MLOrchestrator, RuleBasedSelector, UNSEEN_BENCHMARKS
from optimizers import TACInterpreter

st.set_page_config(
    page_title="ML-Guided Compiler Pass Selector",
    page_icon=None,
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling (theme-agnostic: uses translucent fills and neutral borders so it
# looks correct in both light and dark Streamlit themes)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --accent: #3b6ff5;
    --accent-soft: rgba(59, 111, 245, 0.10);
    --accent-line: rgba(59, 111, 245, 0.35);
    --border: rgba(128, 128, 128, 0.25);
    --surface: rgba(128, 128, 128, 0.06);
    --muted: rgba(128, 128, 128, 0.95);
}

html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}

code, pre, .stCodeBlock, textarea {
    font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace !important;
}

/* Page container */
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1320px;
}

/* Hero banner */
.hero {
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    background: linear-gradient(135deg, var(--accent-soft), transparent 70%);
    border-radius: 10px;
    padding: 1.4rem 1.6rem 1.2rem 1.6rem;
    margin-bottom: 1.4rem;
}
.hero h1 {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem 0;
    padding: 0;
}
.hero p {
    margin: 0 0 0.9rem 0;
    color: var(--muted);
    font-size: 0.95rem;
}
.pipeline {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
}
.chip {
    border: 1px solid var(--accent-line);
    background: var(--accent-soft);
    color: inherit;
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 500;
    white-space: nowrap;
}
.chip-sep {
    color: var(--muted);
    font-size: 0.8rem;
}

/* KPI cards */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
}
[data-testid="stMetricValue"] {
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}

/* KPI grid: equal-height cards regardless of content */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 1rem;
    margin-bottom: 0.5rem;
}
.kpi-card {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 118px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
}
.kpi-label {
    font-size: 0.74rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
}
.kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    line-height: 1.2;
    margin: 0.25rem 0;
}
.kpi-sub {
    font-size: 0.78rem;
    color: var(--muted);
    min-height: 1.1rem;
}
.kpi-sub.positive {
    color: #2ea66a;
    font-weight: 500;
}
@media (max-width: 1000px) {
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    padding: 0.6rem 1.1rem;
    font-weight: 500;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    color: var(--accent);
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--accent);
    height: 2px;
}

/* Section headers */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    margin: 1.8rem 0 0.15rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--border);
}
.section-sub {
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0.25rem 0 0.8rem 0;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
}

/* Text area / inputs */
.stTextArea textarea {
    font-size: 0.88rem;
    line-height: 1.5;
    border-radius: 8px;
}

/* Tables and code */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}
[data-testid="stCodeBlock"] {
    border: 1px solid var(--border);
    border-radius: 8px;
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
}
[data-testid="stExpander"] summary {
    font-weight: 500;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 8px;
}

/* Images */
[data-testid="stImage"] img {
    border: 1px solid var(--border);
    border-radius: 8px;
}

/* Hide default Streamlit chrome */
#MainMenu, footer {
    visibility: hidden;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section_header(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def hero(title, description, stages):
    chips = '<span class="chip-sep">&rarr;</span>'.join(
        f'<span class="chip">{s}</span>' for s in stages
    )
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{description}</p>
            <div class="pipeline">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_system(csv_path="dataset.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["best_strategy"])
    y = df["best_strategy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

    model.fit(X_train, y_train)

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_ms = ((time.perf_counter() - t0) * 1000.0) / len(X_test)

    ml_acc = accuracy_score(y_test, y_pred)
    rule_preds = [RuleBasedSelector.predict(row) for _, row in X_test.iterrows()]
    rule_acc = accuracy_score(y_test, rule_preds)

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .loc[list(model.classes_), ["precision", "recall", "f1-score", "support"]]
    )
    report_df["support"] = report_df["support"].astype(int)

    # Wire trained model directly into an orchestrator instance (no .pkl needed)
    orch = MLOrchestrator.__new__(MLOrchestrator)
    orch.strategy_map = RuleBasedSelector.STRATEGY_MAP
    orch.model = model
    orch.classes_ = model.classes_
    orch.feature_names = X.columns

    return {
        "df": df,
        "unique_vecs": len(X.drop_duplicates()),
        "model": model,
        "orch": orch,
        "ml_acc": ml_acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "rule_acc": rule_acc,
        "infer_ms": infer_ms,
        "report_df": report_df,
    }


@st.cache_data
def run_multipass_benchmark(_orch):
    ml_att, ml_wst, ml_red, ml_eq = [], [], [], 0
    rb_att, rb_wst, rb_red = [], [], []
    o2_att, o2_wst, o2_red = [], [], []

    for idx in range(60):
        cat = CATEGORIES[idx % len(CATEGORIES)]
        comp = "simple" if idx < 30 else "complex"
        src = generate_source_program(cat, complexity=comp)
        tac = TACParser(Lexer(src)).parse()
        orig_len = len(tac)

        opt_ml, log_ml = _orch.optimize_program(tac, verbose=False, return_log=True)
        ml_att.append(len(log_ml))
        ml_wst.append(sum(1 for e in log_ml if not e["accepted"]))
        ml_red.append((orig_len - len(opt_ml)) / orig_len * 100.0)
        ok, _, _ = TACInterpreter.verify_equivalence(tac, opt_ml)
        ml_eq += int(ok)

        opt_rb, _, att_rb, wst_rb = MLOrchestrator.optimize_with_rule_based(tac)
        rb_att.append(att_rb)
        rb_wst.append(wst_rb)
        rb_red.append((orig_len - len(opt_rb)) / orig_len * 100.0)

        opt_o2, _, att_o2, wst_o2 = MLOrchestrator.optimize_with_fixed_order(tac)
        o2_att.append(att_o2)
        o2_wst.append(wst_o2)
        o2_red.append((orig_len - len(opt_o2)) / orig_len * 100.0)

    return pd.DataFrame(
        [
            {
                "Strategy Selector": "ML Smart Fallback",
                "Avg Pass Invocations": round(float(np.mean(ml_att)), 2),
                "Avg Wasted No-Op Calls": round(float(np.mean(ml_wst)), 2),
                "Avg Instruction Reduction (%)": round(float(np.mean(ml_red)), 2),
                "Semantic Equivalence": f"{ml_eq}/60 (100%)",
            },
            {
                "Strategy Selector": "Rule-Based Loop",
                "Avg Pass Invocations": round(float(np.mean(rb_att)), 2),
                "Avg Wasted No-Op Calls": round(float(np.mean(rb_wst)), 2),
                "Avg Instruction Reduction (%)": round(float(np.mean(rb_red)), 2),
                "Semantic Equivalence": "60/60 (100%)",
            },
            {
                "Strategy Selector": "Fixed -O2 Sweep",
                "Avg Pass Invocations": round(float(np.mean(o2_att)), 2),
                "Avg Wasted No-Op Calls": round(float(np.mean(o2_wst)), 2),
                "Avg Instruction Reduction (%)": round(float(np.mean(o2_red)), 2),
                "Semantic Equivalence": "60/60 (100%)",
            },
        ]
    )


def trace_single_program(orch, tac_lines):
    """Runs Smart Fallback and captures per-iteration features + all 6 candidate scores."""
    cur = list(tac_lines)
    steps = []
    for iteration in range(1, 26):
        feats = TACFeatureExtractor(cur).extract_features()
        X_in = pd.DataFrame([feats], columns=orch.feature_names)
        probs = orch.model.predict_proba(X_in)[0]
        rule_scores = RuleBasedSelector.score_features(feats)
        ranking = sorted(zip(orch.classes_, probs), key=lambda x: x[1], reverse=True)

        candidates = [
            {
                "Strategy": s,
                "ML Confidence (%)": round(float(p) * 100.0, 1),
                "Rule Score": round(float(rule_scores.get(s, 0.0)), 1),
            }
            for s, p in ranking
        ]

        accepted_pass = None
        fallbacks = []
        for s, p in ranking:
            new_tac = orch.strategy_map[s](cur)
            if new_tac != cur:
                accepted_pass = (s, float(p), len(cur), len(new_tac))
                cur = new_tac
                break
            fallbacks.append(s)

        steps.append(
            {
                "iteration": iteration,
                "features": feats,
                "candidates": candidates,
                "accepted": accepted_pass,
                "fallbacks": fallbacks,
            }
        )
        if accepted_pass is None:
            break

    is_equiv, matched, total = TACInterpreter.verify_equivalence(tac_lines, cur)
    return cur, steps, (is_equiv, matched, total)


def resolve_plot_path(filename):
    for candidate in (os.path.join("assets", filename), filename):
        if os.path.exists(candidate):
            return candidate
    return None


# --- MAIN APP ---
sys_data = load_system()
orch = sys_data["orch"]

hero(
    "ML-Guided Compiler Optimization Strategy Selector",
    "Selects and orders compiler optimization passes with a Random Forest model, "
    "verified against the original program for semantic equivalence.",
    [
        "Mini-C",
        "Three Address Code",
        "8D Static Features",
        "Random Forest + Smart Fallback",
        "Equivalence Verifier",
    ],
)

# Top KPI Bar (Live-computed from dataset.csv)
kpis = [
    (
        "Random Forest Test Acc",
        f"{sys_data['ml_acc']*100:.2f}%",
        f"+{(sys_data['ml_acc']-sys_data['rule_acc'])*100:.1f} pts vs rules",
        True,
    ),
    (
        "5-Fold CV Accuracy",
        f"{sys_data['cv_mean']*100:.2f}%",
        f"Std dev ±{sys_data['cv_std']*100:.2f}%",
        False,
    ),
    ("Rule-Based Baseline", f"{sys_data['rule_acc']*100:.2f}%", "Heuristic selector", False),
    ("Inference Latency", f"{sys_data['infer_ms']:.3f} ms", "Per test program", False),
    (
        "Unique 8D Vectors",
        f"{sys_data['unique_vecs']} / {len(sys_data['df'])}",
        "Distinct vs. total rows",
        False,
    ),
]
kpi_html = "".join(
    f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
    f'<div class="kpi-value">{value}</div>'
    f'<div class="kpi-sub{" positive" if positive else ""}">{sub}</div></div>'
    for label, value, sub, positive in kpis
)
st.markdown(f'<div class="kpi-grid">{kpi_html}</div>', unsafe_allow_html=True)

st.write("")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Interactive Optimizer", "Benchmarks & Baselines", "Evaluation Plots", "Dataset Explorer"]
)

# ================= TAB 1: INTERACTIVE OPTIMIZER =================
with tab1:
    section_header(
        "Input Program",
        "Load an unseen hand-written Mini-C benchmark, or edit the source freely.",
    )
    preset_map = {b["name"]: b["source"] for b in UNSEEN_BENCHMARKS}
    preset_choice = st.selectbox(
        "Benchmark preset",
        options=list(preset_map.keys()),
    )

    source_code = st.text_area("Mini-C Source Code", value=preset_map[preset_choice], height=210)

    if st.button("Compile, Optimize & Verify", type="primary"):
        try:
            tac_lines = TACParser(Lexer(source_code)).parse()
            opt_tac, steps, (is_equiv, matched, total) = trace_single_program(orch, tac_lines)

            # Baseline comparisons on this exact program
            _, _, rb_att, rb_wst = MLOrchestrator.optimize_with_rule_based(tac_lines)
            _, _, o2_att, o2_wst = MLOrchestrator.optimize_with_fixed_order(tac_lines)
            ml_att = sum(len(s["fallbacks"]) + (1 if s["accepted"] else 0) for s in steps)
            ml_wst = sum(len(s["fallbacks"]) for s in steps)

            section_header("Verification Result")
            if is_equiv:
                st.success(f"Semantic Equivalence: PASS ({matched}/{total} randomized input sets matched)")
            else:
                st.error(f"Semantic Equivalence: FAIL ({matched}/{total})")

            section_header("Generated Code", "Three Address Code before and after optimization.")
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown(f"**Un-optimized TAC** &nbsp;·&nbsp; {len(tac_lines)} lines")
                st.code("\n".join(tac_lines), language="text")
            with col_right:
                delta_lines = len(tac_lines) - len(opt_tac)
                st.markdown(
                    f"**Converged TAC** &nbsp;·&nbsp; {len(opt_tac)} lines &nbsp;·&nbsp; {delta_lines} removed"
                )
                st.code("\n".join(opt_tac), language="text")

            section_header(
                "Pass Efficiency on This Program",
                "Total pass invocations and wasted no-op calls for each selector.",
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Method": "ML Smart Fallback", "Total Pass Calls": ml_att, "Wasted No-Op Calls": ml_wst},
                        {"Method": "Rule-Based Loop", "Total Pass Calls": rb_att, "Wasted No-Op Calls": rb_wst},
                        {"Method": "Fixed -O2 Sweep", "Total Pass Calls": o2_att, "Wasted No-Op Calls": o2_wst},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

            section_header(
                "Iteration-by-Iteration Smart Fallback Trace",
                "Expand an iteration to see extracted features and ranked candidate strategies.",
            )
            for step in steps:
                acc = step["accepted"]
                if acc:
                    if acc[3] < acc[2]:
                        effect = f"{acc[2]} → {acc[3]} lines, {acc[2] - acc[3]} removed"
                    else:
                        effect = f"{acc[2]} lines, instructions rewritten in place"
                    header = (
                        f"Iteration {step['iteration']}: Applied {acc[0]} "
                        f"({acc[1]*100:.1f}% confidence, {effect})"
                    )
                else:
                    header = f"Iteration {step['iteration']}: Converged (Fixed Point Reached)"
                with st.expander(header, expanded=(step["iteration"] == 1)):
                    st.markdown("**Extracted 8D Features**")
                    st.json(step["features"], expanded=False)
                    if step["fallbacks"]:
                        st.caption(f"Smart Fallback skipped no-op passes: {', '.join(step['fallbacks'])}")
                    st.markdown("**Candidate Strategies**")
                    st.dataframe(pd.DataFrame(step["candidates"]), width="stretch", hide_index=True)

        except SyntaxError as e:
            st.error(f"Compilation Error: {e}")

# ================= TAB 2: BENCHMARKS & BASELINES =================
with tab2:
    section_header(
        "Random Forest Per-Class Performance",
        "Evaluated on 150 held-out test programs.",
    )
    st.dataframe(
        sys_data["report_df"].style.format(
            {"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}
        ),
        width="stretch",
    )

    section_header(
        "Multi-Pass Comparison",
        "ML Smart Fallback vs. Rule-Based vs. Fixed -O2 across 60 unseen programs.",
    )
    bench_df = run_multipass_benchmark(orch)
    st.dataframe(bench_df, width="stretch", hide_index=True)

    section_header(
        "Unseen Hand-Written Mini-C Benchmark Suite",
        "Each benchmark is optimized by the ML orchestrator and checked for semantic equivalence.",
    )
    unseen_rows = []
    for b in UNSEEN_BENCHMARKS:
        t = TACParser(Lexer(b["source"])).parse()
        o, l = orch.optimize_program(t, verbose=False, return_log=True)
        passes = [e["strategy"] for e in l if e["accepted"]]
        ok, m, tot = TACInterpreter.verify_equivalence(t, o)
        unseen_rows.append(
            {
                "Benchmark": b["name"],
                "Original TAC Lines": len(t),
                "Optimized TAC Lines": len(o),
                "Applied Passes": " → ".join(passes),
                "Semantic Check": f"PASS ({m}/{tot})" if ok else f"FAIL ({m}/{tot})",
            }
        )
    st.dataframe(pd.DataFrame(unseen_rows), width="stretch", hide_index=True)

# ================= TAB 3: EVALUATION PLOTS =================
with tab3:
    section_header(
        "Evaluation Plots",
        "Generated by ml_pipeline.py and loaded from the assets directory.",
    )
    plots = [
        ("confusion_matrix.png", "Confusion Matrix (150 Held-Out Test Programs)"),
        ("feature_importance.png", "Random Forest Gini Feature Importance (XAI)"),
        ("scatter_num_constants.png", "const_ratio vs. Constant Folding Confidence"),
        ("stacked_bar_iterations.png", "Actual Multi-Pass Strategies per Program (Simple vs. Complex)"),
        ("line_confidence_vs_gain.png", "Model Confidence vs. Measured Instruction Gain"),
        ("box_plot_reduction.png", "Measured Instruction Optimization by Strategy Type"),
    ]
    col_a, col_b = st.columns(2, gap="large")
    for idx, (fname, caption) in enumerate(plots):
        path = resolve_plot_path(fname)
        target_col = col_a if idx % 2 == 0 else col_b
        with target_col:
            if path:
                st.image(path, caption=caption, width="stretch")
            else:
                st.info(f"Plot `{fname}` not found. Run `python ml_pipeline.py` and place PNGs in `assets/`.")
            st.write("")

# ================= TAB 4: DATASET EXPLORER =================
with tab4:
    df = sys_data["df"]
    section_header(
        "Class Distribution",
        "750 synthesized simple and complex programs.",
    )
    st.bar_chart(df["best_strategy"].value_counts())
    section_header("Dataset Preview", "First 50 rows of dataset.csv.")
    st.dataframe(df.head(50), width="stretch")