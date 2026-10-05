# ⚽ VisionPlay Pro

### Advanced Football Analytics & Decision Support System

VisionPlay Pro is an AI-powered football video analysis platform that combines computer vision, player tracking, tactical analytics, and a Decision Support System (DSS) to extract meaningful insights from football match footage.

The system processes uploaded football videos using YOLOv8-based tracking and generates annotated videos, player/team analytics, tactical recommendations, and comprehensive match reports.

---

## 🚀 Overview

VisionPlay Pro transforms raw football footage into structured tactical and performance insights.

Instead of simply detecting players, the system analyzes the game situation and provides actionable recommendations such as:

- ⚽ Shooting opportunities
- 🔄 Passing opportunities
- 🏃 Dribbling opportunities
- 🧠 Tactical decision recommendations
- 🎯 Goal probability
- 📊 Player and team statistics
- 🔥 Player heatmaps
- 🧭 Passing-lane analysis
- ⚡ Ball possession
- 🏟️ Game-state awareness

The application provides both an **annotated match video** and a **web-based analytics dashboard**.

---

## ✨ Key Features

### 🎥 AI Video Analysis

Upload football footage and let the system automatically process the video.

**Supported formats:**

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`

**Maximum upload size:** `50 MB`

The system processes videos asynchronously so that the web interface remains responsive while analysis is running.

---

### 👥 Player Detection & Tracking

VisionPlay Pro uses YOLOv8-based computer vision to detect and track players throughout the video.

Tracked players are assigned unique IDs, allowing the system to maintain player identity across frames.

Example:

```text
RP35
RP3
BP23
BP2

🎯 Action Detection
The system supports detection and analysis of three primary football actions:
Action	Description
⚽ Shooting	Detects shooting events and evaluates shot quality
🔄 Passing	Identifies passing opportunities and passing situations
🏃 Dribbling	Identifies situations where dribbling may be beneficial


🧠 Decision Support System
The core feature of VisionPlay Pro is the Tactical Decision Support System (DSS).
The DSS analyzes the current game context and generates tactical recommendations.
Possible recommendations include:
Shoot
Pass
Dribble
Hold
Clear

Example:
Generated 7 tactical recommendations

Shoot       0
Pass        7
Dribble     0

Average Decision Confidence: 72.2%

The goal of the DSS is to move beyond simple object detection toward context-aware tactical analysis.
📊 Analytics
VisionPlay Pro generates several football analytics metrics.
Match Statistics
- Total Goals
- Total Shots
- Total Passes
- Match Duration
- Pass Accuracy
- Shot Accuracy
Advanced Metrics
- Average Shot Quality
- Goal Conversion
- Key Events
- Top Possession Player
Tactical Analytics
- Player Heatmap
- Team Assignment
- Ball Possession
- Passing Lanes
- Game State Awareness
- Goal Probability
🔥 Player Heatmap
The analytics module provides spatial information about player positioning throughout the match.
This can help identify:
- Frequently occupied areas
- Player positioning
- Attacking and defensive zones
- Movement patterns
🔄 Passing Lane Analysis
The system analyzes player positioning and available passing opportunities.
This can help identify:
- Potential passing targets
- Passing opportunities
- Spatial awareness
- Available passing lanes
Example:
Player maintains good spatial awareness with 7 passing opportunities identified.

🎯 Goal Probability & Shot Quality
For detected shooting situations, VisionPlay Pro evaluates the quality of the opportunity.
Example:
Shot Quality: 40.5%
Goal Probability: 41%

These metrics provide additional context around shooting events rather than simply counting shots.
🎬 Annotated Video Output
After processing, VisionPlay Pro produces an annotated version of the uploaded video.
The processed video can contain:
- Player bounding boxes
- Player tracking IDs
- Team identification
- Ball tracking
- Goal probability
- Tactical information
- Match statistics
- Event information
Example:
Goals: 0
Shots: 0
Passes: 0
Time: 3.3s
Goal Probability: 0%

📋 Event Timeline
Important events detected during processing are displayed in an event timeline.
Example:
5.8s    ⚽ Shot by Player 86    41%

This provides a chronological view of important match events.
🖥️ Application Workflow
                    ┌─────────────────┐
                    │   Upload Video  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Select Analysis │
                    │    Options      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ YOLOv8 Tracking │
                    │  & Detection    │
                    └────────┬────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │ Football Event Analysis│
                 │                         │
                 │ • Shooting              │
                 │ • Passing               │
                 │ • Dribbling             │
                 │ • Possession            │
                 └────────────┬────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  DSS Analysis   │
                    │                 │
                    │ Shoot / Pass /  │
                    │ Dribble / Hold  │
                    └────────┬────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │ Analytics JSON    │
                   │ + Processed Video │
                   └────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Analytics Dashboard│
                  │  + Match Report    │
                  └────────────────────┘

🛠️ Technology Stack
Backend
- Python
- Flask
- OpenCV
- Threading
- JSON
Computer Vision
- YOLOv8
- OpenCV
- Object Detection
- Object Tracking
Frontend
- HTML5
- CSS3
- JavaScript
- Flask / Jinja Templates
Analytics
- Tactical Decision Support System
- Player Tracking
- Team Assignment
- Ball Possession
- Passing Analysis
- Shot Analysis
- Goal Probability
- Heatmaps
📁 Project Structure
football_tracker/
│
├── analytics/
│
├── data/
│   └── videos/
│
├── processed/
│
├── templates/
│   ├── index.html
│   ├── index2.html
│   ├── match_report.html
│   ├── match_report_dss.html
│   ├── match2.html
│   ├── res2.html
│   └── result_dss.html
│
├── uploads/
│
├── app.py
├── decision_system.py
├── track_football.py
├── tracker.py
├── yolov8n.pt
├── football-tracker-video.mp4
├── LICENSE
└── README.md

⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/Anand3074/football_tracker.git
cd football_tracker

2. Create a Virtual Environment
This project uses a dedicated Python virtual environment.
python3 -m venv venv

3. Activate the Virtual Environment
macOS / Linux
source venv/bin/activate

Windows
venv\Scripts\activate

After activation, you should see:
(venv)

in your terminal.
4. Upgrade pip
python -m pip install --upgrade pip

5. Install Dependencies
If requirements.txt is available:
pip install -r requirements.txt

Otherwise, install the core dependencies:
pip install flask opencv-python ultralytics

▶️ Running the Application
Make sure the virtual environment is activated:
source venv/bin/activate

Start the application:
python app.py

You should see:
🚀 Starting VisionPlay Football Analytics Server with DSS...
📁 Upload folder: ...
🎬 Processed folder: ...
📊 Analytics folder: ...
🧠 Decision Support System: ENABLED

The application runs on:
http://127.0.0.1:5001

Open the URL in your browser.
🖱️ How to Use
1. Open VisionPlay Pro
Navigate to:
http://127.0.0.1:5001

2. Upload a Football Video
You can either:
- Drag and drop a football video
- Click the upload area and select a video
Supported formats:
MP4
AVI
MOV
MKV

Maximum file size:
50 MB

3. Select Actions
Select the football actions you want to analyze:
- ☑ Shooting
- ☑ Passing
- ☑ Dribbling
4. Select Analytics
Available analytics include:
- ☑ Player Heatmap
- ☑ Team Assignment
- ☑ Ball Possession
5. Enable DSS Features
The Decision Support System provides:
- ☑ Shooting Recommendations
- ☑ Passing Lane Analysis
- ☑ Dribbling Opportunities
- ☑ Game State Awareness
6. Start Analysis
Click:
🚀 Start Analysis with DSS
The video will be uploaded and processed in the background.
7. View Processing Results
After processing, the application displays:
🎬 Processed Video
The processed video contains computer-vision overlays and analysis information.
📊 Match Statistics
View:
- Goals
- Shots
- Passes
- Match duration
- Pass accuracy
- Shot accuracy
🧠 DSS Decision Breakdown
View recommendations such as:
Shoot
Pass
Dribble
Hold
Clear

🎯 Advanced Metrics
View:
Average Shot Quality
Goal Conversion
Key Events
Top Possession

📋 Event Timeline
Review detected football events chronologically.
📑 Match Report
VisionPlay Pro can generate a comprehensive match report from the generated analytics data.
The report consolidates:
- Match statistics
- Tactical decisions
- DSS insights
- Player information
- Event information
- Analytical metrics
🔌 Application Endpoints
Endpoint	Method	Purpose
/	GET / POST	Main application
/upload	POST	Upload video
/status/<job_id>	GET	Check processing status
/video/<filename>	GET	Serve processed video
/result/<filename>	GET	Display analysis results
/analytics/<filename>	GET	Serve analytics JSON
/match_report/<analytics_filename>	GET	Generate match report
/cleanup	GET / POST	Clean generated files


🧹 Cleanup
The application provides a cleanup endpoint for removing generated files.
It clears files from:
uploads/
processed/
analytics/

and resets the current processing status.
📦 Generated Files
Each analysis creates unique output files.
uploads/
└── <job_id>_<original_video>.mp4

processed/
└── processed_<job_id>_<video>.mp4

analytics/
└── analytics_<job_id>.json

Unique job IDs prevent different uploads from overwriting each other.
🧠 Architecture
                    Flask Web Application
                              │
                              ▼
                        Video Upload
                              │
                              ▼
                    Background Processing
                              │
                              ▼
                    YOLOv8 Detection
                       & Tracking
                              │
                              ▼
                    Football Analytics
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
              Shooting     Passing      Dribbling
                 │            │            │
                 └────────────┼────────────┘
                              ▼
                   Decision Support System
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
              Shoot          Pass         Dribble
                              │
                              ▼
                       Analytics JSON
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Processed Video      Web Dashboard
                                        │
                                        ▼
                                  Match Report

📸 Screenshots
VisionPlay Pro Dashboard
 
Video Analysis
 
DSS Analytics
 
Match Report
 
🎥 Demo
A sample football video is included in the repository:
football-tracker-video.mp4

Start the application and upload the sample video through the VisionPlay Pro interface.
🔐 Security Notes
This project is currently intended for local/development usage.
Before deploying publicly, consider:
- Moving the Flask secret key to an environment variable
- Adding authentication
- Adding CSRF protection
- Validating uploaded files
- Restricting upload size
- Running Flask behind a production WSGI server
- Adding access control to generated videos and analytics
- Adding persistent job storage
⚠️ Current Limitations
- Processing speed depends on available hardware
- Tracking accuracy depends on video quality
- Camera angle can affect player detection
- Occlusion can affect tracking
- Short clips may produce limited statistics
- Tactical recommendations depend on detected positional information
- Current upload limit is 50 MB
🔮 Future Improvements
- [ ] Real-time live match analysis
- [ ] GPU acceleration
- [ ] Persistent database for match analytics
- [ ] Player profiles
- [ ] Team performance comparison
- [ ] Advanced possession models
- [ ] Pass success prediction
- [ ] Expected Goals (xG)
- [ ] Expected Assists (xA)
- [ ] Player heatmap comparison
- [ ] Tactical formation detection
- [ ] Offside detection
- [ ] Automated match summaries
- [ ] Multi-camera support
- [ ] Cloud deployment
- [ ] REST API
- [ ] User authentication
- [ ] Match history dashboard
👨‍💻 Author
Anand
Software Engineer | Full Stack Developer | AI & Computer Vision Enthusiast
📄 License
This project is licensed under the MIT License.
See LICENSE for details.
⭐ Support
If you find this project useful or interesting, consider giving the repository a ⭐.

### One important thing

Since this is specifically for **GitHub**, save it exactly as:

```text
README.md

not .txt, .html, or .markdown.
