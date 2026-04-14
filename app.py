import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import random
import time
from collections import defaultdict

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deadlock Prevention & Recovery Toolkit",
    page_icon="🔒",
    layout="wide"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2a4a 50%, #1e3a5f 100%);
        padding: 2rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .safe-box {
        background: #052e16;
        border-left: 5px solid #2e7d32;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 0.5rem 0;
    }
    .danger-box {
        background: #3f1d1d;
        border-left: 5px solid #c62828;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #1e293b;
        border-left: 5px solid #1565c0;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Banker's Algorithm ────────────────────────────────────────────────────────
def bankers_algorithm(allocation, max_need, available):
    """
    Returns (is_safe, safe_sequence, steps_log).
    allocation : n_processes × n_resources
    max_need   : n_processes × n_resources
    available  : 1D array of n_resources
    """
    n = len(allocation)
    r = len(available)

    need = [[max_need[i][j] - allocation[i][j] for j in range(r)] for i in range(n)]
    work = list(available)
    finish = [False] * n
    safe_seq = []
    steps = []

    for _ in range(n):
        found = False
        for i in range(n):
            if not finish[i] and all(need[i][j] <= work[j] for j in range(r)):
                steps.append({
                    "Process": f"P{i}",
                    "Need": need[i][:],
                    "Work Before": work[:],
                    "Action": "✅ Selected & Executed",
                })
                for j in range(r):
                    work[j] += allocation[i][j]
                finish[i] = True
                safe_seq.append(f"P{i}")
                found = True
                break
        if not found:
            break

    is_safe = all(finish)
    return is_safe, safe_seq, steps, need

# ── Deadlock Detection (Resource Allocation Graph) ────────────────────────────
def detect_deadlock_rag(processes, resources, assignments, requests):
    """
    Simple cycle detection on the RAG.
    assignments : dict {process: [resources held]}
    requests    : dict {process: [resources requested]}
    Returns (deadlocked_processes, graph_edges)
    """
    # Build adjacency: P→R (request), R→P (assignment)
    adj = defaultdict(list)
    edges = []

    for proc, res_list in assignments.items():
        for res in res_list:
            adj[res].append(proc)          # R → P (assignment edge)
            edges.append((res, proc, "assign"))

    for proc, res_list in requests.items():
        for res in res_list:
            adj[proc].append(res)          # P → R (request edge)
            edges.append((proc, res, "request"))

    # DFS cycle detection
    visited = set()
    rec_stack = set()
    deadlocked = set()

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                deadlocked.add(node)
                deadlocked.add(neighbor)
        rec_stack.discard(node)

    all_nodes = list(processes) + list(resources)
    for node in all_nodes:
        if node not in visited:
            dfs(node)

    return list(deadlocked), edges

# ── Wait-For Graph Deadlock Detection ────────────────────────────────────────
def detect_deadlock_wfg(wait_for):
    """
    wait_for: dict {process: [processes it is waiting for]}
    Returns (has_deadlock, deadlocked_processes, cycle_path)
    """
    visited = {}   # 0=unvisited, 1=in-progress, 2=done
    cycle_path = []

    def dfs(node, path):
        visited[node] = 1
        path.append(node)
        for neighbor in wait_for.get(node, []):
            if neighbor not in visited or visited[neighbor] == 0:
                if dfs(neighbor, path):
                    return True
            elif visited[neighbor] == 1:
                # Found cycle
                idx = path.index(neighbor)
                cycle_path.extend(path[idx:])
                return True
        visited[node] = 2
        path.pop()
        return False

    for node in wait_for:
        if node not in visited or visited[node] == 0:
            if dfs(node, []):
                break

    all_procs = set(wait_for.keys()) | {p for lst in wait_for.values() for p in lst}
    deadlocked = list(set(cycle_path)) if cycle_path else []
    return bool(deadlocked), deadlocked, cycle_path

# ── Recovery: Process Termination ────────────────────────────────────────────
def recover_by_termination(deadlocked_processes, priority_order=None):
    """Terminate processes one by one until deadlock is resolved."""
    if priority_order:
        to_terminate = [p for p in priority_order if p in deadlocked_processes]
    else:
        to_terminate = list(deadlocked_processes)
    steps = []
    remaining = list(deadlocked_processes)
    for proc in to_terminate:
        if proc in remaining:
            remaining.remove(proc)
            steps.append({
                "Step": len(steps) + 1,
                "Terminated": proc,
                "Remaining Deadlocked": list(remaining),
                "Status": "🔓 Deadlock Resolved" if not remaining else "🔄 Continuing..."
            })
            if not remaining:
                break
    return steps

# ── Recovery: Resource Preemption ────────────────────────────────────────────
def recover_by_preemption(deadlocked_processes, assignments):
    """Preempt resources from victim processes."""
    steps = []
    freed = []
    remaining = list(deadlocked_processes)
    for proc in list(deadlocked_processes):
        if proc in assignments:
            res = assignments[proc]
            freed.extend(res)
            remaining.remove(proc)
            steps.append({
                "Step": len(steps) + 1,
                "Victim Process": proc,
                "Preempted Resources": res,
                "Resources Freed": freed[:],
                "Status": "🔓 Resolved" if not remaining else "🔄 Continuing..."
            })
            if not remaining:
                break
    return steps

# ─── Build RAG Plotly Figure ──────────────────────────────────────────────────
def build_rag_figure(processes, resources, assignments, requests, deadlocked):
    """Draw Resource Allocation Graph using Plotly."""
    n_p = len(processes)
    n_r = len(resources)

    node_x, node_y, node_text, node_color, node_symbol = [], [], [], [], []

    # Processes on left column
    proc_pos = {}
    for i, p in enumerate(processes):
        x, y = 0.1, 1 - i / max(n_p, 1)
        proc_pos[p] = (x, y)
        node_x.append(x); node_y.append(y)
        color = "#ef5350" if p in deadlocked else "#42a5f5"
        node_text.append(p)
        node_color.append(color)
        node_symbol.append("circle")

    # Resources on right column
    res_pos = {}
    for i, r in enumerate(resources):
        x, y = 0.9, 1 - i / max(n_r, 1)
        res_pos[r] = (x, y)
        node_x.append(x); node_y.append(y)
        node_text.append(r)
        node_color.append("#66bb6a")
        node_symbol.append("square")

    # Build edges
    edge_traces = []
    all_pos = {**proc_pos, **res_pos}

    for proc, res_list in assignments.items():
        for res in res_list:
            if proc in all_pos and res in all_pos:
                x0, y0 = res_pos[res]
                x1, y1 = proc_pos[proc]
                color = "#ef5350" if proc in deadlocked else "#78909c"
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    showlegend=False
                ))

    for proc, res_list in requests.items():
        for res in res_list:
            if proc in all_pos and res in all_pos:
                x0, y0 = proc_pos[proc]
                x1, y1 = res_pos[res]
                color = "#ef5350" if proc in deadlocked else "#ffa726"
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    showlegend=False
                ))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=[28 if s == "circle" else 24 for s in node_symbol],
            color=node_color,
            symbol=node_symbol,
            line=dict(width=2, color="white")
        ),
        showlegend=False
    )

    layout = go.Layout(
        title="Resource Allocation Graph (RAG)",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=420
    )

    fig = go.Figure(data=edge_traces + [node_trace], layout=layout)

    # Legend annotations
    fig.add_annotation(x=0.1, y=-0.08, xref="paper", yref="paper",
                       text="🔵 Process (Blue=Safe, Red=Deadlocked)", showarrow=False, font_size=11)
    fig.add_annotation(x=0.7, y=-0.08, xref="paper", yref="paper",
                       text="🟩 Resource  ──Assign  ···Request", showarrow=False, font_size=11)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🔒 Deadlock Prevention & Recovery Toolkit</h1>
    <p style="font-size:1.1rem;opacity:0.9;">Banker's Algorithm • RAG Detection • Wait-For Graph • Recovery Strategies</p>
</div>
""", unsafe_allow_html=True)

# Dashboard Stats
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("🧠 Algorithms", "3", "Banker's, RAG, WFG")
with c2: st.metric("🛡️ Prevention", "Active", "Banker's Safe-State")
with c3: st.metric("🔍 Detection", "Active", "Cycle Detection")
with c4: st.metric("🔧 Recovery", "Active", "Terminate / Preempt")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧠 Banker's Algorithm",
    "📊 RAG Detection",
    "🔄 Wait-For Graph",
    "🔧 Recovery Strategies",
    "🎲 Deadlock Simulator"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BANKER'S ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🧠 Banker's Algorithm — Deadlock Prevention")
    st.markdown("""
    <div class="info-box">
    <b>How it works:</b> The Banker's Algorithm checks if the system is in a <b>safe state</b>
    before granting any resource request. A safe state means there exists at least one
    <b>safe sequence</b> to complete all processes without deadlock.
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### ⚙️ System Configuration")
        n_proc = st.slider("Number of Processes", 2, 6, 4, key="ba_proc")
        n_res = st.slider("Number of Resource Types", 2, 4, 3, key="ba_res")

        # Quick-fill presets
        preset = st.selectbox("Load Example Preset", [
            "Custom Input",
            "Classic Banker's (Safe)",
            "Deadlock Scenario (Unsafe)"
        ], key="ba_preset")

        if preset == "Classic Banker's (Safe)" and n_proc >= 4 and n_res >= 3:
            alloc_default = [[0,1,0],[2,0,0],[3,0,2],[2,1,1]][:n_proc]
            max_default   = [[7,5,3],[3,2,2],[9,0,2],[2,2,2]][:n_proc]
            avail_default = [3,3,2][:n_res]
        elif preset == "Deadlock Scenario (Unsafe)":
            alloc_default = [[1,0],[0,1],[1,0]][:n_proc]
            max_default   = [[2,1],[1,2],[2,1]][:n_proc]
            avail_default = [0,0][:n_res]
        else:
            alloc_default = [[0]*n_res for _ in range(n_proc)]
            max_default   = [[1]*n_res for _ in range(n_proc)]
            avail_default = [2]*n_res

    with col_right:
        st.markdown("#### 📥 Available Resources")
        avail_cols = st.columns(n_res)
        available = []
        for j in range(n_res):
            v = avail_cols[j].number_input(
                f"R{j}", min_value=0, max_value=20,
                value=avail_default[j] if j < len(avail_default) else 2,
                key=f"avail_{j}"
            )
            available.append(v)

    st.markdown("#### 📋 Allocation Matrix")
    alloc_cols = st.columns(n_res + 1)
    alloc_cols[0].markdown("**Process**")
    for j in range(n_res): alloc_cols[j+1].markdown(f"**R{j}**")

    allocation = []
    for i in range(n_proc):
        row_cols = st.columns(n_res + 1)
        row_cols[0].markdown(f"**P{i}**")
        row = []
        for j in range(n_res):
            v = row_cols[j+1].number_input(
                "", min_value=0, max_value=20,
                value=alloc_default[i][j] if i < len(alloc_default) and j < len(alloc_default[i]) else 0,
                key=f"alloc_{i}_{j}", label_visibility="collapsed"
            )
            row.append(v)
        allocation.append(row)

    st.markdown("#### 📋 Maximum Need Matrix")
    max_cols = st.columns(n_res + 1)
    max_cols[0].markdown("**Process**")
    for j in range(n_res): max_cols[j+1].markdown(f"**R{j}**")

    max_need = []
    for i in range(n_proc):
        row_cols = st.columns(n_res + 1)
        row_cols[0].markdown(f"**P{i}**")
        row = []
        for j in range(n_res):
            v = row_cols[j+1].number_input(
                "", min_value=0, max_value=20,
                value=max_default[i][j] if i < len(max_default) and j < len(max_default[i]) else 2,
                key=f"max_{i}_{j}", label_visibility="collapsed"
            )
            row.append(v)
        max_need.append(row)

    if st.button("▶️ Run Banker's Algorithm", key="run_bankers"):
        # Validate: allocation ≤ max_need
        valid = True
        for i in range(n_proc):
            for j in range(n_res):
                if allocation[i][j] > max_need[i][j]:
                    st.error(f"❌ Allocation[P{i}][R{j}] = {allocation[i][j]} > Max[P{i}][R{j}] = {max_need[i][j]}. Invalid!")
                    valid = False
                    break

        if valid:
            is_safe, safe_seq, steps, need = bankers_algorithm(allocation, max_need, list(available))

            st.markdown("---")
            if is_safe:
                st.markdown(f"""
                <div class="safe-box">
                <h3>✅ System is in a SAFE STATE</h3>
                <p><b>Safe Sequence:</b> {" → ".join(safe_seq)}</p>
                <p>The Banker's Algorithm found a valid execution order. No deadlock will occur.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="danger-box">
                <h3>❌ System is in an UNSAFE STATE</h3>
                <p>No safe sequence exists. The system may enter a DEADLOCK if resource requests proceed.</p>
                </div>
                """, unsafe_allow_html=True)

            # Need matrix display
            st.markdown("#### 📊 Computed Need Matrix (Max − Allocation)")
            need_df = pd.DataFrame(need, columns=[f"R{j}" for j in range(n_res)],
                                   index=[f"P{i}" for i in range(n_proc)])
            st.dataframe(need_df)

            # Steps log
            if steps:
                st.markdown("#### 🔍 Execution Steps")
                steps_df = pd.DataFrame(steps)
                st.dataframe(steps_df, use_container_width=True)

            # Bar chart of allocation vs need
            fig = go.Figure()
            for j in range(n_res):
                fig.add_trace(go.Bar(
                    name=f"Allocation R{j}",
                    x=[f"P{i}" for i in range(n_proc)],
                    y=[allocation[i][j] for i in range(n_proc)],
                    marker_color=px.colors.qualitative.Safe[j % 10]
                ))
                fig.add_trace(go.Bar(
                    name=f"Need R{j}",
                    x=[f"P{i}" for i in range(n_proc)],
                    y=[need[i][j] for i in range(n_proc)],
                    marker_color=px.colors.qualitative.Pastel[j % 10]
                ))
            fig.update_layout(
                title="Allocation vs Need per Process",
                barmode="group",
                xaxis_title="Process",
                yaxis_title="Units"
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RAG DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Resource Allocation Graph (RAG) — Deadlock Detection")
    st.markdown("""
    <div class="info-box">
    <b>RAG:</b> Processes are circles, resources are squares.
    An edge P→R means process P is <b>requesting</b> R.
    An edge R→P means resource R is <b>assigned</b> to P.
    A <b>cycle</b> in the RAG indicates a deadlock.
    </div>
    """, unsafe_allow_html=True)

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        rag_procs = st.number_input("Number of Processes", 2, 6, 4, key="rag_p")
        rag_res = st.number_input("Number of Resources", 2, 5, 3, key="rag_r")
        processes_list = [f"P{i}" for i in range(int(rag_procs))]
        resources_list = [f"R{i}" for i in range(int(rag_res))]

    with r_col2:
        st.markdown("#### Example Scenarios")
        rag_preset = st.selectbox("Load Preset", [
            "Custom",
            "No Deadlock (Safe)",
            "Deadlock (Cycle Exists)"
        ], key="rag_preset")

    st.markdown("#### 🔗 Assignments (Resource → Process)")
    st.caption("Which resources are currently held by which processes?")
    assignments = {}
    for p in processes_list:
        held = st.multiselect(f"{p} holds:", resources_list, key=f"assign_{p}")
        if held:
            assignments[p] = held

    st.markdown("#### 🔁 Requests (Process → Resource)")
    st.caption("Which resources is each process waiting for?")

    # Apply preset defaults
    if rag_preset == "No Deadlock (Safe)" and int(rag_procs) >= 3:
        req_defaults = {"P0": ["R1"], "P1": [], "P2": ["R2"]}
    elif rag_preset == "Deadlock (Cycle Exists)" and int(rag_procs) >= 3:
        req_defaults = {"P0": ["R1"], "P1": ["R2"], "P2": ["R0"]}
    else:
        req_defaults = {}

    requests_rag = {}
    for p in processes_list:
        default_req = req_defaults.get(p, [])
        waiting = st.multiselect(f"{p} is waiting for:", resources_list,
                                  default=default_req, key=f"req_{p}")
        if waiting:
            requests_rag[p] = waiting

    if st.button("🔍 Detect Deadlock (RAG)", key="detect_rag"):
        deadlocked, edges = detect_deadlock_rag(
            processes_list, resources_list, assignments, requests_rag
        )

        if deadlocked:
            st.markdown(f"""
            <div class="danger-box">
            <h3>⚠️ DEADLOCK DETECTED!</h3>
            <p><b>Deadlocked Processes:</b> {', '.join(deadlocked)}</p>
            <p>A cycle was found in the Resource Allocation Graph.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="safe-box">
            <h3>✅ No Deadlock Detected</h3>
            <p>No cycle found in the Resource Allocation Graph. System is safe.</p>
            </div>
            """, unsafe_allow_html=True)

        fig = build_rag_figure(
            processes_list, resources_list,
            assignments, requests_rag, deadlocked
        )
        st.plotly_chart(fig, use_container_width=True)

        # Edge table
        if edges:
            edge_df = pd.DataFrame(edges, columns=["From", "To", "Type"])
            edge_df["Type"] = edge_df["Type"].map({"assign": "R→P Assignment", "request": "P→R Request"})
            st.dataframe(edge_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WAIT-FOR GRAPH
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔄 Wait-For Graph — Cycle-Based Deadlock Detection")
    st.markdown("""
    <div class="info-box">
    <b>Wait-For Graph (WFG):</b> A simplified version of RAG where only processes appear.
    An edge P<sub>i</sub> → P<sub>j</sub> means P<sub>i</sub> is waiting for a resource held by P<sub>j</sub>.
    A <b>cycle</b> = <b>Deadlock</b>.
    </div>
    """, unsafe_allow_html=True)

    w_col1, w_col2 = st.columns([1, 1])
    with w_col1:
        wfg_procs = st.slider("Number of Processes", 3, 7, 4, key="wfg_p")
        proc_names = [f"P{i}" for i in range(wfg_procs)]

        wfg_preset = st.selectbox("Load Preset", [
            "Custom",
            "No Deadlock",
            "2-Process Deadlock",
            "3-Process Deadlock Cycle"
        ], key="wfg_preset")

        if wfg_preset == "No Deadlock":
            preset_wfg = {"P0": ["P1"], "P1": ["P2"], "P2": [], "P3": []}
        elif wfg_preset == "2-Process Deadlock":
            preset_wfg = {"P0": ["P1"], "P1": ["P0"], "P2": [], "P3": []}
        elif wfg_preset == "3-Process Deadlock Cycle":
            preset_wfg = {"P0": ["P1"], "P1": ["P2"], "P2": ["P0"], "P3": []}
        else:
            preset_wfg = {}

    with w_col2:
        st.markdown("#### Build Wait-For Edges")
        wait_for = {}
        for p in proc_names:
            others = [q for q in proc_names if q != p]
            defaults = preset_wfg.get(p, [])
            defaults = [d for d in defaults if d in others]
            waiting = st.multiselect(f"{p} waits for:", others, default=defaults, key=f"wfg_{p}")
            wait_for[p] = waiting

    if st.button("🔄 Detect Deadlock (WFG)", key="detect_wfg"):
        has_dl, deadlocked_wfg, cycle = detect_deadlock_wfg(wait_for)

        if has_dl:
            st.markdown(f"""
            <div class="danger-box">
            <h3>⚠️ DEADLOCK DETECTED via Wait-For Graph!</h3>
            <p><b>Cycle Found:</b> {" → ".join(cycle)} → {cycle[0] if cycle else ""}</p>
            <p><b>Deadlocked Processes:</b> {', '.join(deadlocked_wfg)}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="safe-box">
            <h3>✅ No Deadlock — No Cycle in Wait-For Graph</h3>
            </div>
            """, unsafe_allow_html=True)

        # Plot WFG
        node_x, node_y, node_text, node_color = [], [], [], []
        angle_step = 2 * 3.14159 / max(len(proc_names), 1)
        pos = {}
        for i, p in enumerate(proc_names):
            x = 0.5 + 0.4 * np.cos(i * angle_step)
            y = 0.5 + 0.4 * np.sin(i * angle_step)
            pos[p] = (x, y)
            node_x.append(x); node_y.append(y)
            node_text.append(p)
            node_color.append("#ef5350" if p in deadlocked_wfg else "#42a5f5")

        edge_traces = []
        for p, waiting_list in wait_for.items():
            for q in waiting_list:
                if p in pos and q in pos:
                    x0, y0 = pos[p]
                    x1, y1 = pos[q]
                    color = "#ef5350" if p in deadlocked_wfg and q in deadlocked_wfg else "#ffa726"
                    edge_traces.append(go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None],
                        mode="lines",
                        line=dict(color=color, width=2.5),
                        showlegend=False
                    ))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            marker=dict(size=36, color=node_color, line=dict(width=2, color="white")),
            showlegend=False
        )

        fig_wfg = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title="Wait-For Graph",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="#f8f9fa",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
        )
        st.plotly_chart(fig_wfg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RECOVERY STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔧 Deadlock Recovery Strategies")

    st.markdown("""
    <div class="info-box">
    Once a deadlock is detected, the system must recover. Two main strategies are:
    <ul>
    <li><b>Process Termination:</b> Kill one or all deadlocked processes to release resources.</li>
    <li><b>Resource Preemption:</b> Forcefully take resources from a victim process and give them to others.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown("#### Setup Deadlock Scenario")
        rec_procs = st.slider("Deadlocked Processes Count", 2, 6, 3, key="rec_p")
        deadlocked_set = [f"P{i}" for i in range(rec_procs)]
        st.markdown(f"**Deadlocked:** {', '.join(deadlocked_set)}")

        # Assign resources held
        st.markdown("#### Resources Held by Each Process")
        rec_assignments = {}
        all_res_names = [f"R{i}" for i in range(5)]
        for p in deadlocked_set:
            held = st.multiselect(f"{p} holds:", all_res_names, key=f"rec_assign_{p}",
                                   default=[all_res_names[deadlocked_set.index(p) % len(all_res_names)]])
            rec_assignments[p] = held

    with rec_col2:
        st.markdown("#### Recovery Configuration")
        recovery_method = st.radio(
            "Select Recovery Method",
            ["🔴 Process Termination", "🟡 Resource Preemption", "Compare Both"],
            key="rec_method"
        )

        termination_order = st.multiselect(
            "Termination Priority Order (first = highest priority to kill)",
            deadlocked_set,
            default=deadlocked_set,
            key="rec_order"
        )

    if st.button("🔧 Execute Recovery", key="run_recovery"):
        if "Termination" in recovery_method or "Compare" in recovery_method:
            st.markdown("### 🔴 Process Termination Recovery")
            term_steps = recover_by_termination(deadlocked_set, termination_order)
            if term_steps:
                df_term = pd.DataFrame(term_steps)
                st.dataframe(df_term, use_container_width=True)
                # Progress chart
                fig_t = go.Figure(go.Bar(
                    x=[f"Step {s['Step']}" for s in term_steps],
                    y=[len(s["Remaining Deadlocked"]) for s in term_steps],
                    marker_color=["#ef5350" if s["Remaining Deadlocked"] else "#66bb6a" for s in term_steps],
                    text=[f"Killed: {s['Terminated']}" for s in term_steps],
                    textposition="auto"
                ))
                fig_t.update_layout(title="Remaining Deadlocked Processes After Each Termination",
                                    yaxis_title="Count", xaxis_title="Step")
                st.plotly_chart(fig_t, use_container_width=True)

        if "Preemption" in recovery_method or "Compare" in recovery_method:
            st.markdown("### 🟡 Resource Preemption Recovery")
            pre_steps = recover_by_preemption(deadlocked_set, rec_assignments)
            if pre_steps:
                df_pre = pd.DataFrame(pre_steps)
                st.dataframe(df_pre, use_container_width=True)
                # Freed resources chart
                fig_p = go.Figure(go.Bar(
                    x=[f"Step {s['Step']}" for s in pre_steps],
                    y=[len(s["Resources Freed"]) for s in pre_steps],
                    marker_color="#ffa726",
                    text=[f"Freed: {s['Preempted Resources']}" for s in pre_steps],
                    textposition="auto"
                ))
                fig_p.update_layout(title="Cumulative Resources Freed After Each Preemption",
                                    yaxis_title="Resources Freed", xaxis_title="Step")
                st.plotly_chart(fig_p, use_container_width=True)

        if "Compare" in recovery_method:
            st.markdown("### ⚖️ Strategy Comparison")
            cmp_df = pd.DataFrame({
                "Metric": ["Steps to Resolve", "Processes Killed", "Resources Freed", "System Impact"],
                "Process Termination": [
                    len(recover_by_termination(deadlocked_set, termination_order)),
                    len(deadlocked_set),
                    "None",
                    "High (data loss)"
                ],
                "Resource Preemption": [
                    len(recover_by_preemption(deadlocked_set, rec_assignments)),
                    "0 (if rollback)",
                    str(sum(len(v) for v in rec_assignments.values())),
                    "Medium (rollback cost)"
                ]
            })
            st.table(cmp_df)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DEADLOCK SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🎲 Deadlock Scenario Simulator")
    st.markdown("""
    <div class="info-box">
    Generate and simulate custom deadlock scenarios. Watch how different system states
    lead to deadlock or safe execution. Use random generation to stress-test your understanding.
    </div>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        sim_procs = st.slider("Processes", 3, 8, 4, key="sim_p")
        sim_res = st.slider("Resource Types", 2, 5, 3, key="sim_r")
        sim_mode = st.radio("Scenario Type", [
            "🎲 Random Safe Scenario",
            "💀 Force Deadlock Scenario",
            "🔀 Borderline Scenario"
        ], key="sim_mode")
        total_each = st.slider("Total units per resource type", 3, 15, 7, key="sim_total")

    with sim_col2:
        st.markdown("#### 📝 Simulation Info")
        st.markdown(f"""
        - **Processes:** P0 to P{sim_procs-1}
        - **Resources:** R0 to R{sim_res-1}
        - **Total per Resource:** {total_each} units
        - **Mode:** {sim_mode}
        """)

    if st.button("🎲 Generate & Simulate", key="run_sim"):
        random.seed(42 if "Deadlock" in sim_mode else random.randint(0, 9999))

        if "Safe" in sim_mode:
            # Generate safe scenario
            avail_sim = [total_each // 2] * sim_res
            alloc_sim = [[random.randint(0, 2) for _ in range(sim_res)] for _ in range(sim_procs)]
            max_sim = [[alloc_sim[i][j] + random.randint(1, 3) for j in range(sim_res)] for i in range(sim_procs)]

        elif "Deadlock" in sim_mode:
            # Force unsafe: zero available, circular allocation
            avail_sim = [0] * sim_res
            alloc_sim = [[1 if j == i % sim_res else 0 for j in range(sim_res)] for i in range(sim_procs)]
            max_sim = [[2 if j == i % sim_res else 0 for j in range(sim_res)] for i in range(sim_procs)]

        else:
            # Borderline: available just barely enough
            alloc_sim = [[random.randint(0, 2) for _ in range(sim_res)] for _ in range(sim_procs)]
            max_sim = [[alloc_sim[i][j] + random.randint(0, 2) for j in range(sim_res)] for i in range(sim_procs)]
            avail_sim = [1] * sim_res

        is_safe_sim, seq_sim, steps_sim, need_sim = bankers_algorithm(alloc_sim, max_sim, avail_sim[:])

        st.markdown("### 📊 Generated Scenario")
        proc_labels = [f"P{i}" for i in range(sim_procs)]
        res_labels = [f"R{j}" for j in range(sim_res)]

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Allocation**")
            st.dataframe(pd.DataFrame(alloc_sim, index=proc_labels, columns=res_labels))
        with col_b:
            st.markdown("**Max Need**")
            st.dataframe(pd.DataFrame(max_sim, index=proc_labels, columns=res_labels))
        with col_c:
            st.markdown("**Available**")
            st.dataframe(pd.DataFrame([avail_sim], columns=res_labels))

        st.markdown("### 🧠 Banker's Algorithm Result")
        if is_safe_sim:
            st.markdown(f"""
            <div class="safe-box">
            <h3>✅ SAFE STATE — No Deadlock</h3>
            <p><b>Safe Sequence:</b> {" → ".join(seq_sim)}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="danger-box">
            <h3>❌ UNSAFE STATE — Deadlock Risk!</h3>
            <p>No safe sequence found. Recovery needed.</p>
            </div>
            """, unsafe_allow_html=True)

        # Heatmap of Need
        fig_heat = go.Figure(data=go.Heatmap(
            z=need_sim,
            x=res_labels,
            y=proc_labels,
            colorscale="RdYlGn_r",
            text=need_sim,
            texttemplate="%{text}",
            showscale=True
        ))
        fig_heat.update_layout(title="Need Matrix Heatmap", height=350)
        st.plotly_chart(fig_heat, use_container_width=True)

        if steps_sim:
            st.markdown("### 🔍 Execution Log")
            st.dataframe(pd.DataFrame(steps_sim), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<center>
<small>🔒 Deadlock Prevention & Recovery Toolkit | Built with Python + Streamlit + Plotly<br>
OS Concepts: Banker's Algorithm | Resource Allocation Graph | Wait-For Graph | Recovery</small>
</center>
""", unsafe_allow_html=True)