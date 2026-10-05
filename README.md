# ⚽ VisionPlay Pro

**Football video analytics with a tactical Decision Support System (DSS).**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

VisionPlay Pro takes a football clip, detects and tracks players and the ball with YOLOv8, detects passes, shots and goals, and runs a rule-based Decision Support System that recommends the best tactical option (shoot, pass or dribble) for the player in possession. Results are delivered as an annotated video, a JSON analytics file, a web dashboard and a match report.

---

## 📑 Table of Contents

- [Quick Start](#-quick-start)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Using the App](#-using-the-app)
- [Required Files](#-required-files)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Decision Support System](#-decision-support-system)
- [API Endpoints](#-api-endpoints)
- [Generated Files](#-generated-files)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Limitations](#-limitations)
- [Roadmap](#-roadmap)
- [Security Notes](#-security-notes)
- [Author](#-author)
- [License](#-license)

---

## ⚡ Quick Start

```bash
git clone https://github.com/Anand3074/football_tracker.git
cd football_tracker
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5001** in your browser and upload a football clip.

---

## 📋 Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.9 or newer (3.10+ recommended) |
| pip | Latest version recommended |
| Git | To clone the repository |
| Disk space | ~1 GB (PyTorch and Ultralytics dependencies) |
| Hardware | CPU is enough; a CUDA GPU speeds up processing |

### Python libraries

All dependencies are listed in [`requirements.txt`](requirements.txt):

| Library | Purpose |
|---|---|
| `flask` | Web server, routing, templates |
| `opencv-python` | Video reading/writing, frame annotation |
| `ultralytics` | YOLOv8 detection and ByteTrack tracking (installs PyTorch) |
| `numpy` | Numerical computations |

The remaining imports (`json`, `threading`, `uuid`, `collections`, `dataclasses`, `enum`, etc.) are part of the Python standard library.

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Anand3074/football_tracker.git
cd football_tracker
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate it

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows (PowerShell / CMD)**
```bash
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal prompt.

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. YOLOv8 weights

The app loads `yolov8n.pt` from the project root. The file is included in the repository; if it is missing, Ultralytics downloads it automatically on first run (internet required).

---

## ▶️ Running the Application

Make sure the virtual environment is active, then:

```bash
python app.py
```

Expected console output:

```text
🚀 Starting VisionPlay Football Analytics Server with DSS...
📁 Upload folder: .../uploads
🎬 Processed folder: .../processed
📊 Analytics folder: .../analytics
🧠 Decision Support System: ENABLED
```

The server runs at:

```text
http://127.0.0.1:5001
```

Stop it with `Ctrl + C`. Deactivate the virtual environment with `deactivate`.

---

## 🖱 Using the App

1. **Upload** a video (drag and drop or click to browse). Supported: `.mp4`, `.avi`, `.mov`, `.mkv`, up to **50 MB**.
2. **Select actions** to analyze: Shooting, Passing, Dribbling.
3. **Select analytics**: Player Heatmap, Team Assignment, Ball Possession.
4. **Enable DSS features**: Shooting Recommendations, Passing Lane Analysis, Dribbling Opportunities, Game State Awareness.
5. Click **🚀 Start Analysis with DSS**. Processing runs in the background and the page polls `/status/<job_id>` for progress.
6. **Review results**: processed video, match statistics, DSS decision breakdown, advanced metrics and the event timeline.
7. Open the **Match Report** for a consolidated summary.

> **Note:** to keep processing fast, only the **first 10 seconds** of each uploaded video are analyzed (see [Configuration](#-configuration)).

---

## 📁 Required Files

Everything needed to run the project:

```text
football_tracker/
│
├── app.py                    # Flask app: routes, upload, background jobs
├── track_football.py         # Video pipeline, analytics, JSON + match report
├── tracker.py                # Tracking, possession, event detection, rendering
├── decision_system.py        # Decision Support System (tactical engine)
├── requirements.txt          # Python dependencies
├── yolov8n.pt                # YOLOv8 nano weights (auto-downloaded if missing)
│
├── templates/
│   ├── index.html            # Upload page
│   ├── result_dss.html       # Results dashboard with DSS insights
│   └── match_report_dss.html # Match report
│
├── football-tracker-video.mp4  # Sample clip for testing
├── LICENSE
└── README.md
```

Created automatically at runtime: `uploads/`, `processed/`, `analytics/` and a temporary `temp/` folder.

The `templates/` folder also contains earlier template versions (`index2.html`, `match_report.html`, `match2.html`, `res2.html`) that the current app does not use.

| File | Responsibility |
|---|---|
| `app.py` | HTTP routes, file validation, background threads, status tracking |
| `track_football.py` | Orchestrates detection → tracking → events → DSS → output; saves analytics JSON |
| `tracker.py` | `StableTracker`, `BallPossessionTracker`, `EventDetector`, `VisualRenderer` |
| `decision_system.py` | `DecisionEngine`, `SpatialAnalyzer`, `GameState`, `DecisionVisualizer` |

---

## ✨ Features

### Computer vision
- YOLOv8 person and ball detection with ByteTrack tracking
- Persistent player IDs across frames (labels like `RP35`, `BP23`)
- Ball tracking with velocity and direction estimation
- Team assignment (Red / Blue) based on field position

### Football event detection
- **Passes**: detected from possession-transition patterns
- **Shots**: detected from ball speed, direction and field zone
- **Goals**: detected when the ball crosses the goal region
- **Ball possession**: nearest-player proximity model

### Decision Support System
- Per-frame tactical recommendations: **Shoot / Pass / Dribble**
- Confidence, risk and reward scoring for each option
- Passing-lane analysis with interception checks
- Game-state awareness (field phase, score, critical moments)
- Goal probability and shot quality estimation

### Outputs
- 🎬 Annotated video: bounding boxes, IDs, teams, ball, goal probability bar, DSS panel, passing lanes, live stats
- 📊 Analytics JSON with events, decisions and summary
- 🖥 Web dashboard with statistics and event timeline
- 📑 Match report page

### Example output

```text
Generated 7 tactical recommendations
Shoot 0 | Pass 7 | Dribble 0
Average decision confidence: 72.2%

5.8s   ⚽ Shot by Player 86   41%
```

---

## 🧠 How It Works

```text
Upload Video
     │
     ▼
Trim to first 10 seconds
     │
     ▼
YOLOv8 + ByteTrack (players, ball)
     │
     ▼
Possession & Event Detection (pass / shot / goal)
     │
     ▼
Decision Support System (Shoot / Pass / Dribble)
     │
     ▼
Annotated Video  +  Analytics JSON
     │
     ▼
Web Dashboard  →  Match Report
```

Processing runs in a background thread, so the web interface stays responsive during analysis.

---

## 🎯 Decision Support System

For each frame where a player is in possession, the engine scores the available options with weighted factors:

| Action | Factors |
|---|---|
| **Shoot** | Position (35%), angle (25%), defender pressure (20%), game state (15%), confidence (5%) |
| **Pass** | Safety (30%), progression (30%), opportunity (25%), game state (15%) |
| **Dribble** | Space (35%), support (25%), pressure (25%), game state (15%) |

The top 3 options are returned with confidence, risk level, reward potential and a short reasoning list. The pitch is divided into tactical zones (defensive, midfield, attacking, danger zone) to inform the scoring. `Hold` and `Clear` are defined in the action set for future extension.

---

## 🔌 API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET / POST | Main page / form fallback |
| `/upload` | POST | Upload video and start processing (returns `job_id`) |
| `/status/<job_id>` | GET | Processing status and progress |
| `/video/<filename>` | GET | Serve processed video |
| `/result/<filename>?analytics=<file>` | GET | Results dashboard |
| `/analytics/<filename>` | GET | Raw analytics JSON |
| `/match_report/<analytics_filename>` | GET | Match report |
| `/cleanup` | GET / POST | Delete all generated files and reset status |

Example upload response:

```json
{
  "job_id": "…",
  "message": "Video uploaded successfully. Processing started with DSS...",
  "output_filename": "processed_<job_id>_<video>.mp4",
  "analytics_filename": "analytics_<job_id>.json"
}
```

---

## 📦 Generated Files

Each job gets a unique ID, so uploads never overwrite each other.

```text
uploads/    <job_id>_<original_video>.mp4        (deleted after processing)
processed/  processed_<job_id>_<video>.mp4
analytics/  analytics_<job_id>.json
```

Use `/cleanup` to clear `uploads/`, `processed/` and `analytics/`.

---

## ⚙️ Configuration

| Setting | Location | Default |
|---|---|---|
| Server port | `app.py` (`app.run`) | `5001` |
| Max upload size | `app.py` (`MAX_CONTENT_LENGTH`) | 50 MB |
| Allowed formats | `app.py` (`ALLOWED_EXTENSIONS`) | mp4, avi, mov, mkv |
| Analyzed duration | `track_football.py` (`trim_video`, `max_duration`) | 10 s |
| Detection confidence | `track_football.py` (`CONFIDENCE_THRESHOLD`) | 0.3 |
| Possession distance | `track_football.py` (`proximity_threshold`) | 80 px |
| YOLO model | `track_football.py` (`YOLO('yolov8n.pt')`) | YOLOv8 nano |

For better accuracy, swap in a larger model such as `yolov8s.pt` or `yolov8m.pt`.

---

## 🩺 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt` |
| Processed video is empty or will not play in the browser | Your OpenCV build may lack the `avc1` (H.264) codec. Try `mp4v` in `cv2.VideoWriter_fourcc` or re-encode with FFmpeg |
| Port 5001 already in use | Change the port in `app.py` |
| Upload rejected | File must be mp4/avi/mov/mkv and under 50 MB |
| Slow processing | Use a shorter clip, a GPU, or the `yolov8n.pt` model |
| ByteTrack dependency error | Run `pip install lap` |

---

## ⚠️ Limitations

- Only the first 10 seconds of a video are analyzed
- Team assignment uses field position (left/right of midfield), not jersey colour
- Tracking quality depends on video quality, camera angle and occlusion
- Pass accuracy is currently fixed at 100% (no pass-success model yet)
- Match report values for advanced metrics (distance, speed, sprints, duels, etc.) and the opposing team's stats are **simulated placeholders**, not measured from the video
- Job status is stored in memory and resets when the server restarts
- Designed for broadcast-style, single-camera footage

---

## 🔮 Roadmap

- [ ] Real player heatmaps from tracked positions
- [ ] Jersey-colour team classification
- [ ] Pass success prediction and real pass accuracy
- [ ] Expected Goals (xG) and Expected Assists (xA)
- [ ] Replace simulated report metrics with measured ones
- [ ] Tactical formation and offside detection
- [ ] GPU acceleration and real-time live analysis
- [ ] Database for match history and player profiles
- [ ] REST API and user authentication
- [ ] Cloud deployment

---

## 🔐 Security Notes

This project is intended for local development. Before deploying publicly:

- Move `app.secret_key` to an environment variable
- Run with `debug=False` behind a production WSGI server (Gunicorn, Waitress)
- Add authentication, CSRF protection and access control for videos and analytics
- Validate uploaded file contents, not just extensions
- Protect or remove the `/cleanup` endpoint
- Use persistent job storage

---

## 👨‍💻 Author

**Anand**
Software Engineer | Full Stack Developer | AI & Computer Vision Enthusiast

GitHub: [@Anand3074](https://github.com/Anand3074)

---

## 📄 License

Released under the [MIT License](LICENSE).

---

⭐ If you find this project useful, consider giving the repository a star.
