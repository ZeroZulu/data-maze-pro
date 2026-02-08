"""
🌀 Data Maze Pro — Advanced Pathfinding Game & Analytics Platform
=================================================================
A gamified pathfinding visualization tool for comparing algorithms.
Built for Data Science portfolios & interactive learning.

Algorithms: Dijkstra, A*, BFS, Greedy Best-First, Bidirectional BFS
Tech: Streamlit, NumPy, Pandas, Plotly
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import heapq
import time
import math
import io
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Data Maze Pro — Pathfinding Analytics",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    /* Header */
    .maze-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        position: relative;
        overflow: hidden;
    }
    .maze-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(108,92,231,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .maze-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .maze-header p {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.55);
        margin: 0;
        letter-spacing: 0.5px;
    }

    /* Metric Cards */
    .metric-row {
        display: flex;
        gap: 12px;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(108,92,231,0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.45);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #6c5ce7;
    }
    .metric-card .sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: rgba(255,255,255,0.35);
        margin-top: 2px;
    }

    /* Achievement Badge */
    .achievement-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        color: #fff;
        padding: 4px 12px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        margin: 3px 4px;
        font-weight: 600;
    }
    .achievement-locked {
        background: rgba(255,255,255,0.06);
        color: rgba(255,255,255,0.25);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #a29bfe;
    }

    /* Score display */
    .score-display {
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        padding: 12px 20px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
    }
    .score-display .score-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
    }
    .score-display .score-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #fff;
    }

    /* Table styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        margin: 1rem 0;
    }
    .styled-table th {
        background: rgba(108,92,231,0.15);
        color: #a29bfe;
        padding: 10px 14px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid rgba(108,92,231,0.3);
    }
    .styled-table td {
        padding: 8px 14px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: rgba(255,255,255,0.75);
    }
    .styled-table tr:hover td {
        background: rgba(108,92,231,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
if "results_history" not in st.session_state:
    st.session_state.results_history = []
if "total_games" not in st.session_state:
    st.session_state.total_games = 0
if "high_score" not in st.session_state:
    st.session_state.high_score = 0
if "achievements" not in st.session_state:
    st.session_state.achievements = set()

# ─────────────────────────────────────────────
# Maze Generation
# ─────────────────────────────────────────────
def generate_maze(size, seed=None):
    """Generate a weighted grid maze with terrain costs."""
    if seed is not None:
        np.random.seed(seed)

    # Create terrain: 1=open, higher=harder
    grid = np.ones((size, size), dtype=float)

    # Add random walls (impassable)
    wall_pct = 0.20 + np.random.random() * 0.10
    walls = np.random.random((size, size)) < wall_pct
    grid[walls] = 0  # 0 = wall

    # Add weighted terrain
    terrain_mask = (grid == 1) & (np.random.random((size, size)) < 0.35)
    grid[terrain_mask] = np.random.choice([2, 3, 4, 5], size=terrain_mask.sum())

    # Ensure start and end are open
    grid[0, 0] = 1
    grid[size - 1, size - 1] = 1

    # Ensure path exists by carving a basic path
    _ensure_path(grid, size)

    return grid


def _ensure_path(grid, size):
    """Carve a basic path to guarantee solvability."""
    x, y = 0, 0
    np.random.seed(None)
    while x < size - 1 or y < size - 1:
        if grid[x, y] == 0:
            grid[x, y] = 1
        if x < size - 1 and y < size - 1:
            if np.random.random() < 0.5:
                x += 1
            else:
                y += 1
        elif x < size - 1:
            x += 1
        else:
            y += 1
    grid[size - 1, size - 1] = 1


def calculate_difficulty(grid):
    """Calculate maze difficulty 1-10."""
    size = grid.shape[0]
    wall_ratio = np.sum(grid == 0) / grid.size
    avg_cost = np.mean(grid[grid > 0])
    high_cost_ratio = np.sum(grid >= 4) / max(np.sum(grid > 0), 1)
    raw = (wall_ratio * 4 + (avg_cost - 1) / 4 * 3 + high_cost_ratio * 3) + (size / 50) * 2
    return max(1, min(10, round(raw, 1)))

# ─────────────────────────────────────────────
# Pathfinding Algorithms
# ─────────────────────────────────────────────
def get_neighbors(grid, pos):
    """Get valid neighbors (4-directional)."""
    rows, cols = grid.shape
    r, c = pos
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] > 0:
            neighbors.append((nr, nc))
    return neighbors


def heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def dijkstra(grid, start, end):
    """Dijkstra's algorithm — guaranteed optimal."""
    t0 = time.perf_counter()
    dist = {start: 0}
    prev = {start: None}
    explored = set()
    heap = [(0, start)]
    nodes_explored = 0

    while heap:
        cost, current = heapq.heappop(heap)
        if current in explored:
            continue
        explored.add(current)
        nodes_explored += 1

        if current == end:
            break

        for nb in get_neighbors(grid, current):
            new_cost = cost + grid[nb[0], nb[1]]
            if nb not in dist or new_cost < dist[nb]:
                dist[nb] = new_cost
                prev[nb] = current
                heapq.heappush(heap, (new_cost, nb))

    path = _reconstruct(prev, end)
    elapsed = (time.perf_counter() - t0) * 1000
    return path, list(explored), elapsed, nodes_explored, dist.get(end, float('inf'))


def astar(grid, start, end):
    """A* search — optimal with heuristic."""
    t0 = time.perf_counter()
    g = {start: 0}
    prev = {start: None}
    explored = set()
    heap = [(heuristic(start, end), 0, start)]
    nodes_explored = 0

    while heap:
        f, cost, current = heapq.heappop(heap)
        if current in explored:
            continue
        explored.add(current)
        nodes_explored += 1

        if current == end:
            break

        for nb in get_neighbors(grid, current):
            new_g = cost + grid[nb[0], nb[1]]
            if nb not in g or new_g < g[nb]:
                g[nb] = new_g
                f_val = new_g + heuristic(nb, end)
                prev[nb] = current
                heapq.heappush(heap, (f_val, new_g, nb))

    path = _reconstruct(prev, end)
    elapsed = (time.perf_counter() - t0) * 1000
    return path, list(explored), elapsed, nodes_explored, g.get(end, float('inf'))


def bfs(grid, start, end):
    """Breadth-First Search — ignores weights."""
    t0 = time.perf_counter()
    prev = {start: None}
    explored = set([start])
    queue = deque([start])
    nodes_explored = 0

    while queue:
        current = queue.popleft()
        nodes_explored += 1

        if current == end:
            break

        for nb in get_neighbors(grid, current):
            if nb not in explored:
                explored.add(nb)
                prev[nb] = current
                queue.append(nb)

    path = _reconstruct(prev, end)
    elapsed = (time.perf_counter() - t0) * 1000
    # Calculate actual path cost
    cost = sum(grid[p[0], p[1]] for p in path) if path else float('inf')
    return path, list(explored), elapsed, nodes_explored, cost


def greedy_best_first(grid, start, end):
    """Greedy Best-First — fast but non-optimal."""
    t0 = time.perf_counter()
    prev = {start: None}
    explored = set()
    heap = [(heuristic(start, end), start)]
    nodes_explored = 0

    while heap:
        _, current = heapq.heappop(heap)
        if current in explored:
            continue
        explored.add(current)
        nodes_explored += 1

        if current == end:
            break

        for nb in get_neighbors(grid, current):
            if nb not in explored:
                prev[nb] = current
                heapq.heappush(heap, (heuristic(nb, end), nb))

    path = _reconstruct(prev, end)
    elapsed = (time.perf_counter() - t0) * 1000
    cost = sum(grid[p[0], p[1]] for p in path) if path else float('inf')
    return path, list(explored), elapsed, nodes_explored, cost


def bidirectional_bfs(grid, start, end):
    """Bidirectional BFS — searches from both ends."""
    t0 = time.perf_counter()
    explored_fwd = {start: None}
    explored_bwd = {end: None}
    queue_fwd = deque([start])
    queue_bwd = deque([end])
    nodes_explored = 0
    meeting = None

    while queue_fwd or queue_bwd:
        # Forward step
        if queue_fwd:
            current = queue_fwd.popleft()
            nodes_explored += 1
            if current in explored_bwd:
                meeting = current
                break
            for nb in get_neighbors(grid, current):
                if nb not in explored_fwd:
                    explored_fwd[nb] = current
                    queue_fwd.append(nb)

        # Backward step
        if queue_bwd:
            current = queue_bwd.popleft()
            nodes_explored += 1
            if current in explored_fwd:
                meeting = current
                break
            for nb in get_neighbors(grid, current):
                if nb not in explored_bwd:
                    explored_bwd[nb] = current
                    queue_bwd.append(nb)

    # Reconstruct path
    path = []
    if meeting:
        # Forward path
        node = meeting
        fwd_path = []
        while node is not None:
            fwd_path.append(node)
            node = explored_fwd.get(node)
        fwd_path.reverse()

        # Backward path
        node = explored_bwd.get(meeting)
        while node is not None:
            fwd_path.append(node)
            node = explored_bwd.get(node)
        path = fwd_path

    elapsed = (time.perf_counter() - t0) * 1000
    all_explored = list(set(list(explored_fwd.keys()) + list(explored_bwd.keys())))
    cost = sum(grid[p[0], p[1]] for p in path) if path else float('inf')
    return path, all_explored, elapsed, nodes_explored, cost


def _reconstruct(prev, end):
    """Reconstruct path from prev dict."""
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path if len(path) > 1 else []


ALGORITHMS = {
    "Dijkstra": {"fn": dijkstra, "color": "#6c5ce7", "desc": "Guaranteed optimal, systematic exploration", "optimal": True},
    "A*": {"fn": astar, "color": "#00b894", "desc": "Optimal with heuristic guidance", "optimal": True},
    "BFS": {"fn": bfs, "color": "#fdcb6e", "desc": "Unweighted breadth-first search", "optimal": False},
    "Greedy": {"fn": greedy_best_first, "color": "#e17055", "desc": "Fast heuristic-only, non-optimal", "optimal": False},
    "Bidirectional": {"fn": bidirectional_bfs, "color": "#0984e3", "desc": "Search from both ends simultaneously", "optimal": False},
}

# ─────────────────────────────────────────────
# Scoring & Achievements
# ─────────────────────────────────────────────
def calculate_score(path_cost, nodes_explored, time_ms, difficulty, grid_size):
    """Calculate game score based on performance."""
    if path_cost == float('inf') or path_cost == 0:
        return 0
    efficiency = len([]) / max(nodes_explored, 1) * 100  # placeholder
    base = max(0, 1000 - path_cost * 10)
    speed_bonus = max(0, 200 - time_ms * 5)
    explore_bonus = max(0, 300 - nodes_explored * 0.5)
    diff_mult = 1 + (difficulty / 10)
    size_mult = grid_size / 15
    return int((base + speed_bonus + explore_bonus) * diff_mult * size_mult)


def check_achievements(results, difficulty):
    """Check and unlock achievements."""
    new_achievements = set()
    for r in results:
        if r["time_ms"] < 10:
            new_achievements.add("⚡ Speed Demon")
        if r["efficiency"] > 80:
            new_achievements.add("🎯 Efficiency Expert")
        if r["path_cost"] < r["grid_size"] * 1.5:
            new_achievements.add("💰 Cost Optimizer")
    if difficulty >= 7:
        new_achievements.add("🏆 Challenge Master")
    if len(results) >= 5:
        new_achievements.add("📚 Algorithm Scholar")
    return new_achievements

# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def create_maze_visualization(grid, results, selected_algos):
    """Create interactive Plotly maze visualization."""
    size = grid.shape[0]

    fig = go.Figure()

    # Base terrain heatmap
    terrain_colors = grid.copy().astype(float)
    terrain_colors[terrain_colors == 0] = -1  # walls

    fig.add_trace(go.Heatmap(
        z=terrain_colors[::-1],
        colorscale=[
            [0, '#1a1a2e'],     # walls
            [0.2, '#16213e'],   # open
            [0.4, '#1a4a3a'],   # light terrain
            [0.6, '#4a3a1a'],   # medium terrain
            [0.8, '#5a2a1a'],   # heavy terrain
            [1.0, '#6a1a1a'],   # very heavy
        ],
        showscale=False,
        hovertemplate='Row: %{y}<br>Col: %{x}<br>Cost: %{z}<extra></extra>'
    ))

    # Plot explored nodes and paths for each algorithm
    for algo_name in selected_algos:
        if algo_name in results:
            r = results[algo_name]
            color = ALGORITHMS[algo_name]["color"]

            # Explored nodes
            if r["explored"]:
                exp_x = [e[1] for e in r["explored"]]
                exp_y = [size - 1 - e[0] for e in r["explored"]]
                fig.add_trace(go.Scatter(
                    x=exp_x, y=exp_y,
                    mode='markers',
                    marker=dict(size=max(3, 18 - size // 3), color=color, opacity=0.15, symbol='square'),
                    name=f'{algo_name} explored',
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Path
            if r["path"]:
                path_x = [p[1] for p in r["path"]]
                path_y = [size - 1 - p[0] for p in r["path"]]
                fig.add_trace(go.Scatter(
                    x=path_x, y=path_y,
                    mode='lines+markers',
                    line=dict(width=max(2, 6 - size // 15), color=color),
                    marker=dict(size=max(3, 10 - size // 8), color=color),
                    name=f'{algo_name}',
                    hovertemplate=f'{algo_name}<br>Step: %{{text}}<extra></extra>',
                    text=list(range(len(r["path"])))
                ))

    # Start & End markers
    fig.add_trace(go.Scatter(
        x=[0], y=[size - 1],
        mode='markers+text',
        marker=dict(size=16, color='#00b894', symbol='star', line=dict(width=2, color='white')),
        text=['START'], textposition='top center',
        textfont=dict(color='#00b894', size=10, family='JetBrains Mono'),
        name='Start', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[size - 1], y=[0],
        mode='markers+text',
        marker=dict(size=16, color='#e17055', symbol='star', line=dict(width=2, color='white')),
        text=['END'], textposition='bottom center',
        textfont=dict(color='#e17055', size=10, family='JetBrains Mono'),
        name='End', showlegend=False
    ))

    fig.update_layout(
        plot_bgcolor='#0f0c29',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color='rgba(255,255,255,0.7)'),
        margin=dict(l=10, r=10, t=40, b=10),
        height=max(400, min(650, size * 20 + 100)),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.5, size - 0.5]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            scaleanchor='x', range=[-0.5, size - 0.5]
        ),
        legend=dict(
            bgcolor='rgba(15,12,41,0.85)',
            bordercolor='rgba(108,92,231,0.3)',
            borderwidth=1,
            font=dict(size=11)
        ),
        title=dict(
            text='🌀 Maze Visualization',
            font=dict(size=16, family='Space Grotesk'),
            x=0.5
        )
    )

    return fig


def create_radar_chart(results):
    """Create radar chart comparing algorithm metrics."""
    categories = ['Speed', 'Path Quality', 'Efficiency', 'Exploration', 'Reliability']

    fig = go.Figure()

    for algo_name, r in results.items():
        color = ALGORITHMS[algo_name]["color"]
        max_time = max(res["time_ms"] for res in results.values()) or 1
        max_cost = max(res["path_cost"] for res in results.values()) if any(res["path_cost"] < float('inf') for res in results.values()) else 1
        max_nodes = max(res["nodes_explored"] for res in results.values()) or 1

        speed = max(0, 100 - (r["time_ms"] / max_time) * 100) if max_time > 0 else 50
        quality = max(0, 100 - ((r["path_cost"] - min(res["path_cost"] for res in results.values())) / max(max_cost, 1)) * 100) if r["path_cost"] < float('inf') else 0
        efficiency = r["efficiency"]
        exploration = max(0, 100 - (r["nodes_explored"] / max_nodes) * 100) if max_nodes > 0 else 50
        reliability = 100 if ALGORITHMS[algo_name]["optimal"] else 60

        values = [speed, quality, efficiency, exploration, reliability]
        values.append(values[0])  # close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=algo_name,
            line=dict(color=color, width=2),
            fill='toself',
            fillcolor=color.replace(')', ',0.1)').replace('rgb', 'rgba') if 'rgb' in color else f'{color}1a',
            opacity=0.8
        ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(15,12,41,0.5)',
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(108,92,231,0.15)'),
            angularaxis=dict(gridcolor='rgba(108,92,231,0.15)')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color='rgba(255,255,255,0.7)', size=10),
        height=380,
        margin=dict(l=60, r=60, t=40, b=40),
        legend=dict(bgcolor='rgba(15,12,41,0.85)', bordercolor='rgba(108,92,231,0.3)', borderwidth=1),
        title=dict(text='📊 Algorithm Comparison Radar', font=dict(size=14, family='Space Grotesk'), x=0.5)
    )
    return fig


def create_scatter_chart(results):
    """Create efficiency vs speed scatter plot."""
    fig = go.Figure()

    for algo_name, r in results.items():
        color = ALGORITHMS[algo_name]["color"]
        fig.add_trace(go.Scatter(
            x=[r["time_ms"]],
            y=[r["efficiency"]],
            mode='markers+text',
            marker=dict(size=20, color=color, line=dict(width=2, color='white'), opacity=0.9),
            text=[algo_name],
            textposition='top center',
            textfont=dict(size=10, color=color, family='JetBrains Mono'),
            name=algo_name,
            hovertemplate=f'<b>{algo_name}</b><br>Time: %{{x:.2f}}ms<br>Efficiency: %{{y:.1f}}%<extra></extra>'
        ))

    fig.update_layout(
        xaxis=dict(title='Computation Time (ms)', gridcolor='rgba(108,92,231,0.1)'),
        yaxis=dict(title='Efficiency (%)', gridcolor='rgba(108,92,231,0.1)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color='rgba(255,255,255,0.7)', size=10),
        height=350,
        margin=dict(l=60, r=20, t=40, b=50),
        showlegend=False,
        title=dict(text='⚡ Speed vs Efficiency Trade-off', font=dict(size=14, family='Space Grotesk'), x=0.5)
    )
    return fig


def create_bar_chart(results):
    """Create grouped bar chart for key metrics."""
    algos = list(results.keys())
    colors = [ALGORITHMS[a]["color"] for a in algos]

    fig = make_subplots(rows=1, cols=3, subplot_titles=('Path Cost', 'Nodes Explored', 'Time (ms)'))

    costs = [results[a]["path_cost"] if results[a]["path_cost"] < float('inf') else 0 for a in algos]
    nodes = [results[a]["nodes_explored"] for a in algos]
    times = [results[a]["time_ms"] for a in algos]

    fig.add_trace(go.Bar(x=algos, y=costs, marker_color=colors, showlegend=False, hovertemplate='%{x}: %{y:.1f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Bar(x=algos, y=nodes, marker_color=colors, showlegend=False, hovertemplate='%{x}: %{y}<extra></extra>'), row=1, col=2)
    fig.add_trace(go.Bar(x=algos, y=times, marker_color=colors, showlegend=False, hovertemplate='%{x}: %{y:.2f}ms<extra></extra>'), row=1, col=3)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color='rgba(255,255,255,0.7)', size=10),
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(tickangle=45)
    return fig

# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="maze-header">
        <h1>🌀 Data Maze Pro</h1>
        <p>Advanced Pathfinding Game & Analytics Platform · Compare 5 Algorithms · Gamified Learning</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        tab_setup, tab_stats, tab_export = st.tabs(["🎮 Setup", "📊 Stats", "💾 Export"])

        with tab_setup:
            grid_size = st.slider("Grid Size", 8, 50, 15, help="Larger grids = harder mazes")
            seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, help="Same seed = same maze")
            st.markdown("---")
            st.markdown("**Select Algorithms:**")
            selected_algos = []
            for name, info in ALGORITHMS.items():
                if st.checkbox(f"{name}", value=True, help=info["desc"]):
                    selected_algos.append(name)

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                new_maze = st.button("🎲 New Maze", use_container_width=True)
            with col2:
                run_btn = st.button("🚀 Run!", use_container_width=True, type="primary")

        with tab_stats:
            st.markdown(f"**Games Played:** {st.session_state.total_games}")
            st.markdown(f"**High Score:** {st.session_state.high_score:,}")
            st.markdown("---")
            st.markdown("**🏆 Achievements:**")
            all_achievements = ["⚡ Speed Demon", "🎯 Efficiency Expert", "💰 Cost Optimizer", "🏆 Challenge Master", "📚 Algorithm Scholar"]
            for a in all_achievements:
                if a in st.session_state.achievements:
                    st.markdown(f'<span class="achievement-badge">{a}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="achievement-badge achievement-locked">🔒 {a.split(" ", 1)[1]}</span>', unsafe_allow_html=True)

        with tab_export:
            if st.session_state.results_history:
                df = pd.DataFrame(st.session_state.results_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results CSV",
                    csv,
                    f"datamaze_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
                st.markdown(f"*{len(st.session_state.results_history)} records available*")
            else:
                st.info("Run algorithms to generate data!")

    # ── Generate Maze ──
    if new_maze:
        seed = np.random.randint(0, 9999)

    grid = generate_maze(grid_size, seed)
    difficulty = calculate_difficulty(grid)
    start = (0, 0)
    end = (grid_size - 1, grid_size - 1)

    # ── Metrics Row ──
    wall_count = np.sum(grid == 0)
    open_cells = np.sum(grid > 0)
    avg_terrain = np.mean(grid[grid > 0])

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Grid</div>
            <div class="value">{grid_size}×{grid_size}</div>
            <div class="sub">{grid_size**2} cells</div>
        </div>
        <div class="metric-card">
            <div class="label">Difficulty</div>
            <div class="value">{difficulty}/10</div>
            <div class="sub">{"Easy" if difficulty < 3 else "Medium" if difficulty < 6 else "Hard" if difficulty < 8 else "Extreme"}</div>
        </div>
        <div class="metric-card">
            <div class="label">Walls</div>
            <div class="value">{wall_count}</div>
            <div class="sub">{wall_count/grid.size*100:.0f}% blocked</div>
        </div>
        <div class="metric-card">
            <div class="label">Avg Terrain</div>
            <div class="value">{avg_terrain:.1f}</div>
            <div class="sub">cost/cell</div>
        </div>
        <div class="metric-card">
            <div class="label">Seed</div>
            <div class="value">#{seed}</div>
            <div class="sub">reproducible</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Run Algorithms ──
    if run_btn and selected_algos:
        results = {}
        progress = st.progress(0, text="Running algorithms...")

        for i, algo_name in enumerate(selected_algos):
            progress.progress((i + 1) / len(selected_algos), text=f"Running {algo_name}...")
            algo_fn = ALGORITHMS[algo_name]["fn"]
            path, explored, elapsed, nodes_exp, path_cost = algo_fn(grid, start, end)

            path_len = len(path)
            efficiency = (path_len / max(nodes_exp, 1)) * 100 if path_len > 0 else 0

            results[algo_name] = {
                "path": path,
                "explored": explored,
                "path_cost": path_cost,
                "path_length": path_len,
                "nodes_explored": nodes_exp,
                "time_ms": elapsed,
                "efficiency": efficiency,
                "grid_size": grid_size,
            }

            # Save to history
            st.session_state.results_history.append({
                "Timestamp": datetime.now().isoformat(),
                "Algorithm": algo_name,
                "Grid_Size": grid_size,
                "Seed": seed,
                "Difficulty": difficulty,
                "Path_Cost": path_cost,
                "Path_Length": path_len,
                "Nodes_Explored": nodes_exp,
                "Computation_Time_ms": round(elapsed, 3),
                "Efficiency_Pct": round(efficiency, 2),
            })

        progress.empty()
        st.session_state.total_games += 1

        # Check achievements
        result_list = [{"time_ms": r["time_ms"], "efficiency": r["efficiency"],
                        "path_cost": r["path_cost"], "grid_size": grid_size} for r in results.values()]
        new_ach = check_achievements(result_list, difficulty)
        if new_ach - st.session_state.achievements:
            for a in new_ach - st.session_state.achievements:
                st.toast(f"🎉 Achievement Unlocked: {a}")
        st.session_state.achievements |= new_ach

        # ── Maze Visualization ──
        st.plotly_chart(create_maze_visualization(grid, results, selected_algos), use_container_width=True)

        # ── Results Table ──
        st.markdown("### 📋 Algorithm Comparison")
        table_data = []
        for algo_name, r in results.items():
            table_data.append({
                "Algorithm": algo_name,
                "Path Cost": f"{r['path_cost']:.1f}" if r['path_cost'] < float('inf') else "N/A",
                "Path Length": r["path_length"],
                "Nodes Explored": r["nodes_explored"],
                "Time (ms)": f"{r['time_ms']:.2f}",
                "Efficiency": f"{r['efficiency']:.1f}%",
                "Optimal?": "✅" if ALGORITHMS[algo_name]["optimal"] else "❌",
            })

        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        # ── Analytics Dashboard ──
        st.markdown("### 📊 Analytics Dashboard")
        col_radar, col_scatter = st.columns(2)
        with col_radar:
            st.plotly_chart(create_radar_chart(results), use_container_width=True)
        with col_scatter:
            st.plotly_chart(create_scatter_chart(results), use_container_width=True)

        # Bar chart
        st.plotly_chart(create_bar_chart(results), use_container_width=True)

        # ── Score ──
        best_result = min(results.values(), key=lambda r: r["path_cost"])
        best_algo = [k for k, v in results.items() if v == best_result][0]
        score = calculate_score(best_result["path_cost"], best_result["nodes_explored"],
                                best_result["time_ms"], difficulty, grid_size)
        st.session_state.high_score = max(st.session_state.high_score, score)

        st.markdown(f"""
        <div class="score-display">
            <div class="score-label">Round Score</div>
            <div class="score-value">{score:,}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**🏅 Best Algorithm:** {best_algo} — Path Cost: {best_result['path_cost']:.1f}, "
                    f"Time: {best_result['time_ms']:.2f}ms, Efficiency: {best_result['efficiency']:.1f}%")

    elif run_btn and not selected_algos:
        st.warning("⚠️ Select at least one algorithm in the sidebar!")
    else:
        # Show empty maze
        empty_results = {}
        st.plotly_chart(create_maze_visualization(grid, empty_results, []), use_container_width=True)
        st.info("👈 Configure your algorithms in the sidebar and click **🚀 Run!** to start.")

    # ── Footer ──
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.4; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
        Data Maze Pro · Built with Streamlit & Plotly · Pathfinding Algorithm Comparison Tool
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
