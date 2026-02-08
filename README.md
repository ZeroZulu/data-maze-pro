# 🌀 Data Maze Pro — Pathfinding Algorithm Comparison & Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL_HERE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **An interactive pathfinding game and analytics platform that compares 5 algorithms in real-time. Built for data science portfolios to demonstrate algorithmic thinking, data visualization, and performance analysis.**

![Data Maze Pro Screenshot](assets/Screensho.png)

---

## 🎯 What It Does

Data Maze Pro generates weighted grid mazes and runs **5 different pathfinding algorithms** simultaneously, providing comprehensive analytics on their performance. It combines gamification with serious algorithm comparison — making it both fun to play and valuable as a data science demonstration tool.

**Core Data Science Concepts:** Graph Theory · Algorithm Optimization · Performance Benchmarking · Comparative Analysis · Interactive Data Visualization

---

## 🧮 Algorithms Implemented

| Algorithm | Strategy | Optimal? | Best For |
|-----------|----------|----------|----------|
| **Dijkstra** | Systematic cost minimization | ✅ Yes | Guaranteed shortest path |
| **A*** | Heuristic-guided optimal search | ✅ Yes | Fast optimal pathfinding |
| **BFS** | Breadth-first unweighted search | ❌ No | Uniform-cost environments |
| **Greedy Best-First** | Pure heuristic, fastest possible | ❌ No | Quick approximate solutions |
| **Bidirectional BFS** | Search from both ends | ❌ No | Long-distance pathfinding |

---

## 📊 Analytics Features

- **Radar Chart** — Multi-metric algorithm comparison (speed, quality, efficiency, exploration, reliability)
- **Scatter Plot** — Speed vs efficiency trade-off visualization
- **Grouped Bar Charts** — Path cost, nodes explored, and computation time side-by-side
- **Comparison Table** — Detailed performance metrics for each algorithm
- **CSV Export** — Download all results for further analysis in Python, R, or Excel

---

## 🏆 Gamification

- **Scoring System** — Points based on path cost, speed, exploration efficiency, and difficulty
- **5 Achievements** — Speed Demon, Efficiency Expert, Cost Optimizer, Challenge Master, Algorithm Scholar
- **Difficulty Ratings** — Dynamic 1-10 scale based on maze complexity
- **High Score Tracking** — Beat your personal best across sessions

---

## 🚀 Quick Start

### Local Setup (2 minutes)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/data-maze-pro.git
cd data-maze-pro

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run data_maze_pro.py
```

Open **http://localhost:8501** in your browser.

### How to Play

1. Set grid size and random seed in the sidebar
2. Select which algorithms to compare
3. Click **🚀 Run!**
4. Analyze results in the dashboard
5. Export data for further analysis

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core language |
| **Streamlit** | Web framework & UI |
| **NumPy** | Maze generation & array operations |
| **Pandas** | Data management & export |
| **Plotly** | Interactive visualizations |

---

## 📈 Performance Metrics

Each algorithm run produces 12+ metrics:

- **Path Cost** — Total weighted traversal cost
- **Path Length** — Number of steps taken
- **Nodes Explored** — Search space coverage
- **Computation Time** — Algorithm execution speed (ms)
- **Efficiency %** — Path length ÷ nodes explored
- **Difficulty Rating** — Maze complexity score
- **Optimality** — Whether the path is guaranteed shortest

---

## 📂 Project Structure

```
data-maze-pro/
├── data_maze_pro.py       # Main application (800+ lines)
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
├── .streamlit/
│   └── config.toml        # Streamlit theme configuration
├── README.md              # This file
├── DEPLOYMENT_GUIDE.md    # Full deployment instructions
├── DEPLOYMENT_CHECKLIST.md # Step-by-step checklist
└── LICENSE                # MIT License
```

---

## 🌐 Deployment

### Streamlit Community Cloud (Recommended)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **"New app"** → Select this repo → `data_maze_pro.py`
5. Click **Deploy**

Your app will be live at `https://your-app-name.streamlit.app`

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions including Docker and Heroku options.

---

## 🎓 Data Science Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Algorithm Design** | Graph traversal, heuristic optimization, search strategies |
| **Performance Analysis** | Benchmarking, comparative metrics, trade-off analysis |
| **Data Visualization** | Interactive charts, radar plots, scatter analysis |
| **Software Engineering** | Clean architecture, session state, modular design |
| **Communication** | Intuitive UI, clear metrics, exportable results |

---

## 🤝 Part of AI Games Collection

This project is part of a broader portfolio demonstrating AI and data science through interactive games:

| Game | Core Concept |
|------|-------------|
| 🧠 [Tic-Tac-Toe AI](../tictactoe_ai) | Minimax · Alpha-Beta Pruning |
| 🟩 [Wordle Entropy Solver](../wordle_entropy_solver) | Information Theory · Entropy |
| 🌀 **Data Maze Pro** (this) | Pathfinding · Graph Optimization |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with ❤️ for Data Science · Powered by Streamlit & Plotly</i>
</p>
