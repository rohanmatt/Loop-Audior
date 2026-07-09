"""
app.py — The Loop Auditor
Watch an AI grade itself and get better.

Run:  streamlit run app.py
"""

import time
import streamlit as st
import pandas as pd

from backend import (
    LoopEngine,
    RateLimitExhausted,
    HF_BASE_URL,
    HFClient,
    StopConditions,
    Critic,
    RUBRIC,
    single_shot,
    MissingTokenError,
    WRITER_MODEL,
    CRITIC_MODEL,
)

st.set_page_config(page_title="The Loop Auditor", page_icon="◉", layout="wide")

# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 1280px; }
  .lo-hero { font-size: 2.1rem; font-weight: 600; letter-spacing: -.02em; margin-bottom: .1rem; }
  .lo-sub  { color: #7a7a72; font-size: 1rem; margin-bottom: 1.6rem; }
  .lo-card {
      border: 1px solid rgba(128,128,128,.22); border-radius: 12px;
      padding: 1rem 1.15rem; margin-bottom: .75rem;
  }
  .lo-tag {
      display:inline-block; font-size:.7rem; letter-spacing:.08em;
      text-transform:uppercase; padding:.2rem .55rem; border-radius:5px;
      border:1px solid rgba(128,128,128,.3); color:#8a8a82; margin-bottom:.5rem;
  }
  .lo-pass { color:#1D9E75; font-weight:500; }
  .lo-fail { color:#E24B4A; font-weight:500; }
  .lo-mono { font-family: ui-monospace, monospace; font-size:.8rem; color:#8a8a82; }
  .lo-phase {
      display:inline-block; padding:.35rem .8rem; border-radius:20px;
      font-size:.78rem; margin-right:.4rem; border:1px solid rgba(128,128,128,.25);
  }
  .lo-active { background:#7F77DD; color:#fff; border-color:#7F77DD; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
for k, v in {
    "result": None, "events": [], "baseline": None, "running": False
}.items():
    st.session_state.setdefault(k, v)


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ◉ Loop Auditor")
    st.caption("Prompt engineering ships a draft. Loop engineering ships a result.")
    st.divider()

    company = st.text_input("Company", value="Figma")
    focus = st.text_input("Analytical focus", value="AI-native design tooling")

    st.markdown("**Stopping conditions**")
    thresh = st.slider("Pass threshold (avg)", 3.0, 5.0, 4.0, 0.1)
    min_dim = st.slider("Min any dimension", 2.0, 5.0, 4.0, 0.5)
    max_iter = st.slider("Max iterations", 2, 6, 4)

    st.divider()
    run_baseline = st.checkbox("Also run the no-loop control", value=True)
    go = st.button("Run the loop", type="primary", use_container_width=True)

    st.divider()
    client_probe = HFClient()
    if client_probe.available():
        st.caption("🟢 API token loaded from .env")
    else:
        st.caption("🔴 No token — add HF_TOKEN to .env")
    provider = HF_BASE_URL.split("//")[-1].split("/")[0]
    st.caption(f"provider · `{provider}`")
    st.caption(f"writer · `{WRITER_MODEL.split('/')[-1]}`")
    st.caption(f"critic · `{CRITIC_MODEL.split('/')[-1]}`")


st.markdown('<div class="lo-hero">The Loop Auditor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="lo-sub">One model. One prompt. The only variable is whether a loop wraps it.</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
if go:
    st.session_state.events = []
    st.session_state.result = None
    st.session_state.baseline = None

    stop = StopConditions(max_iterations=max_iter, pass_threshold=thresh, min_dimension=min_dim)
    engine = LoopEngine(stop=stop)

    status = st.status("Loop running…", expanded=True)

    def on_event(phase, data):
        st.session_state.events.append({"phase": phase, "data": data, "t": time.strftime("%H:%M:%S")})
        n = data.get("iteration", "")
        labels = {
            "plan": "① PLAN — goal decomposed",
            "execute_start": f"② EXECUTE — writer drafting (iter {n})",
            "execute_done": f"② EXECUTE — draft {n} complete",
            "observe_start": f"③ OBSERVE — critic scoring (iter {n})",
            "observe_done": f"③ OBSERVE — scored {data.get('critique', {}).get('average', '')}/5",
            "reflect": f"④ REFLECT — weakest: {data.get('weakest', '')}",
            "stop": f"⑤ STOP — {data.get('reason', '')}",
            "retry": f"⏳ backing off {data.get('wait')}s — {data.get('why', '')}",
        }
        status.write(labels.get(phase, phase))

    try:
        if run_baseline:
            status.write("◇ CONTROL — single-shot, no loop")
            b_client = HFClient()
            b_draft, b_crit = single_shot(b_client, company, focus)
            st.session_state.baseline = {"draft": b_draft, "critique": b_crit}

        st.session_state.result = engine.run(company, focus, on_event=on_event)
        status.update(label="Loop terminated", state="complete", expanded=False)

    except RateLimitExhausted as e:
        status.update(label="Rate limited", state="error")
        st.error(str(e))
    except MissingTokenError as e:
        status.update(label="Missing credentials", state="error")
        st.error(str(e))
    except Exception as e:
        status.update(label="Loop failed", state="error")
        st.exception(e)


result = st.session_state.result
baseline = st.session_state.baseline


# ---------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------
tab_brief, tab_loop = st.tabs(["  The brief  ", "  The loop  "])


# ============================ TAB 1 =========================================
with tab_brief:
    if not result:
        st.info("Set a company in the sidebar and run the loop.")
        st.markdown(
            "The agent writes a board-level competitive brief. "
            "A separate critic sub-agent grades it against a five-point rubric "
            "and refuses to pass it until it earns the score. You watch that happen."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        curve = result.score_curve
        c1.metric("Iterations", len(result.iterations))
        c2.metric("First score", f"{curve[0]}/5")
        c3.metric("Final score", f"{curve[-1]}/5", delta=f"+{round(curve[-1]-curve[0],2)}")
        c4.metric("Verdict", "PASSED" if result.passed else "BUDGET EXIT",
                  delta=f"{result.total_retries} retries" if result.total_retries else None,
                  delta_color="off")

        st.caption(f"Exit rule fired → {result.stop_reason}")
        st.divider()

        if baseline:
            left, right = st.columns(2)
            with left:
                st.markdown('<span class="lo-tag">Prompt engineering · one shot</span>', unsafe_allow_html=True)
                st.markdown(f"### <span class='lo-fail'>{baseline['critique'].average}/5</span>", unsafe_allow_html=True)
                with st.expander("Read the draft"):
                    st.markdown(baseline["draft"])
                st.caption("Critic's objections:")
                for f in baseline["critique"].failures:
                    st.markdown(f"- {f}")

            with right:
                st.markdown('<span class="lo-tag">Loop engineering · same model</span>', unsafe_allow_html=True)
                cls = "lo-pass" if result.passed else "lo-fail"
                st.markdown(f"### <span class='{cls}'>{curve[-1]}/5</span>", unsafe_allow_html=True)
                st.markdown(result.final_draft)
        else:
            st.markdown(result.final_draft)


# ============================ TAB 2 =========================================
with tab_loop:
    st.markdown("#### The five ingredients, running live")

    if not result:
        st.info("Run the loop to populate this view.")
    else:
        curve = result.score_curve
        phases = ["① PLAN", "② EXECUTE", "③ OBSERVE", "④ REFLECT", "⑤ STOP"]
        st.markdown(
            "".join(f'<span class="lo-phase lo-active">{p}</span>' for p in phases),
            unsafe_allow_html=True,
        )
        st.write("")

        # ---- FEEDBACK CYCLES ------------------------------------------
        st.markdown('<span class="lo-tag">Feedback cycles</span>', unsafe_allow_html=True)
        df = pd.DataFrame(
            {d: [it.critique.scores.get(d, 0) for it in result.iterations] for d in RUBRIC},
            index=[f"iter {it.n}" for it in result.iterations],
        )
        df["average"] = [it.critique.average for it in result.iterations]
        st.line_chart(df[["average"]], height=200)
        st.dataframe(df, use_container_width=True)

        st.divider()
        col_a, col_b = st.columns(2)

        # ---- VERIFICATION ---------------------------------------------
        with col_a:
            st.markdown('<span class="lo-tag">Verification mechanisms</span>', unsafe_allow_html=True)
            st.caption("The one who writes is not the one who checks.")
            for it in result.iterations:
                verdict = "PASS" if it.stopped and result.passed else "REJECTED"
                cls = "lo-pass" if verdict == "PASS" else "lo-fail"
                degraded = getattr(it.critique, "degraded", False)
                suffix = " ⚠ degraded" if degraded else ""
                with st.expander(f"Iteration {it.n} — {it.critique.average}/5 · {verdict}{suffix}",
                                 expanded=(it.n == 1)):
                    if degraded:
                        st.warning("Critic output was unparseable. Scores recovered by pattern match, "
                                   "loop continued. A verifier that fails should not take the loop with it.")
                    st.markdown(f"<span class='{cls}'>{verdict}</span>", unsafe_allow_html=True)
                    for dim, sc in it.critique.scores.items():
                        st.progress(sc / 5, text=f"{dim} — {sc}/5")
                    if it.critique.failures:
                        st.caption("Failures cited:")
                        for f in it.critique.failures:
                            st.markdown(f"- {f}")

        # ---- STOPPING CONDITIONS --------------------------------------
        with col_b:
            st.markdown('<span class="lo-tag">Clear stopping conditions</span>', unsafe_allow_html=True)
            st.caption("A loop without an exit is a bill.")
            for rule in StopConditions(
                max_iterations=max_iter, pass_threshold=thresh, min_dimension=min_dim
            ).describe():
                st.markdown(f"- {rule}")
            st.markdown('<div class="lo-card">', unsafe_allow_html=True)
            st.markdown("**Rule that fired**")
            st.code(result.stop_reason, language=None)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<span class="lo-tag">Automation & tools</span>', unsafe_allow_html=True)
            st.caption("Every model call, logged. Nothing hidden.")
            calls = pd.DataFrame(result.tool_calls)
            if not calls.empty:
                calls["model"] = calls["model"].str.split("/").str[-1]
                cols = [c for c in ["ts", "name", "model", "latency_ms", "retries"] if c in calls]
                st.dataframe(calls[cols],
                             use_container_width=True, hide_index=True)
                note = f" · {result.total_retries} rate-limit retries survived" if result.total_retries else ""
                st.caption(f"{len(calls)} calls · {calls.latency_ms.sum()/1000:.1f}s total{note}")

        st.divider()

        # ---- CONTEXT & MEMORY -----------------------------------------
        st.markdown('<span class="lo-tag">Context & memory</span>', unsafe_allow_html=True)
        st.caption(
            f"The agent forgets between calls. The scratchpad does not. "
            f"Carried forward: {result.memory_size:,} chars."
        )
        for it in result.iterations:
            if it.n == 1:
                continue
            with st.expander(f"What iteration {it.n} was told before it wrote"):
                prev = result.iterations[it.n - 2].critique
                st.markdown(f"**Weakest dimension:** `{prev.weakest}` ({prev.scores[prev.weakest]}/5)")
                st.markdown("**Fix instruction carried into the next draft:**")
                st.info(prev.fix_instruction)

        st.divider()
        st.markdown('<span class="lo-tag">Raw event trace</span>', unsafe_allow_html=True)
        with st.expander("Every phase transition"):
            for e in st.session_state.events:
                st.markdown(
                    f'<span class="lo-mono">{e["t"]} · {e["phase"]}</span>',
                    unsafe_allow_html=True,
                )