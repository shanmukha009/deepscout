"""Streamlit demo shell. BUILD_PLAN: Thread F.

Run: streamlit run app/streamlit_app.py
"""
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
from deepscout.orchestrator import run_research
from deepscout.models import AgentStep

st.set_page_config(page_title="DeepScout", layout="wide")
st.title("DeepScout — Multi-Agent Research Analyst")
st.caption("Plans, searches, reads, verifies claims, and synthesizes a cited report.")

q = st.text_input("Research question", "What is retrieval-augmented generation?")

run_btn = st.button("Run Research", type="primary")

if run_btn and q.strip():
    col_report, col_trace = st.columns([3, 2])

    with col_trace:
        st.subheader("Live Trace")
        trace_container = st.empty()

    with col_report:
        st.subheader("Report")
        report_placeholder = st.empty()
        report_placeholder.info("Agents running…")

    # Accumulate steps for live display
    completed_steps: list[AgentStep] = []

    def _render_trace():
        with trace_container.container():
            for s in completed_steps:
                icon = {
                    "planner": "🗺️",
                    "searcher": "🔍",
                    "reader": "📖",
                    "orchestrator": "🧩",
                    "critic": "🔬",
                    "synthesizer": "✍️",
                }.get(s.agent, "⚙️")
                st.markdown(
                    f"{icon} **{s.agent}** › `{s.action}` &nbsp; "
                    f"_{s.latency_ms:.0f} ms_ &nbsp; "
                    f"${s.cost_usd:.5f}"
                )

    def on_step(step: AgentStep):
        completed_steps.append(step)

    t_start = time.perf_counter()
    report, trace = run_research(q.strip(), step_callback=on_step)
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    _render_trace()

    # Final trace with summary metrics
    with col_trace:
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Steps", len(trace.steps))
        m2.metric("Total cost", f"${trace.total_cost_usd:.5f}")
        m3.metric("Latency", f"{elapsed_ms:.0f} ms")

    # Final report
    with col_report:
        report_placeholder.empty()
        st.markdown(report.to_markdown())

        if report.unanswered:
            with st.expander(f"Dropped / unanswered ({len(report.unanswered)})"):
                for item in report.unanswered:
                    st.markdown(f"- {item}")
