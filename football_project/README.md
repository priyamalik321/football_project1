# Soccer Player Re-Identification - Cross-Camera Player Mapping

##  Objective

Given two video clips of the same soccer game from different camera angles (`broadcast.mp4` and `tacticam.mp4`), the goal is to:
- Detect and track players in each video
- Assign unique IDs to players in both views
- Generate a consistent player mapping across both video feeds using visual/spatial features

---

## Project Structure

computer_vision/
├── football_project/
│ ├── model/
│ │ └── best.pt # YOLOv8 fine-tuned weights
│ ├── src/
│ │ ├── detect.py # Detect players using YOLOv8
│ │ ├── track.py # Track players using DeepSort
│ │ ├── map_players.py # Map players across camera views
│ ├── videos/
│ │ ├── broadcast.mp4
│ │ ├── tacticam.mp4
│ │ ├── broadcast_dets.json # Detections from broadcast.mp4
│ │ ├── tacticam_dets.json # Detections from tacticam.mp4
│ │ ├── broadcast_tracks.json # Tracking output for broadcast.mp4
│ │ ├── tacticam_tracks.json # Tracking output for tacticam.mp4
│ │ └── tacticam_to_broadcast_id_map.json # Final player ID mapping
├── .gitignore
└── README.md



---

## Setup Instructions

1. **Create virtual environment & activate**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # On Windows
Install dependencies


pip install  requirements..
Download Model (if not present)
Get the YOLOv8 trained model from here.

## Running the Pipeline
Run Detection


python football_project/src/detect.py
Run Tracking


python football_project/src/track.py
Run Player Mapping


python football_project/src/map_players.py

## Output
After running all scripts, you will get:

broadcast_tracks.json, tacticam_tracks.json: Player tracks in each video

tacticam_to_broadcast_id_map.json: Mapping of player IDs from tacticam to broadcast view

Example:


{
  "1": 14,
  "2": 19,
  "3": 133
}

## Notes
Object detection is done using a fine-tuned YOLOv8 model.

Player tracking is performed using DeepSort.

Mapping is based on visual feature similarity (e.g., bounding box trajectory/position).





###  .gitignore (Recommended)

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg
*.egg-info/
dist/
build/

# VSCode
.vscode/
*.code-workspace

# Environments
.venv/
.env

# OS
.DS_Store
Thumbs.db

# Video outputs (optional)
*broad_cast.mp4
*tacticam.mp4

# JSON outputs 
football_project/videos/broadcast_dets.json
football_project/videos/tacticam_dets.json
football_project/videos/broadcast_tracks.json
football_project/videos/tacticam_tracks.json
