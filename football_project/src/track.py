import cv2
import json
from deep_sort_realtime.deepsort_tracker import DeepSort
import tqdm
import numpy as np
import pathlib
import os

def run_tracker(video_path, detections_json, output_json):
    # Load detections
    with open(detections_json, 'r') as f:
        detections = json.load(f)

    cap = cv2.VideoCapture(video_path)
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

    tracks_output = []
    frame_id = 0
    pbar = tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    while True:
        ret, frame = cap.read()
        if not ret or frame_id >= len(detections):
            break

        frame_dets = detections[frame_id]
        boxes = np.array(frame_dets['boxes'])
        scores = np.array(frame_dets['scores'])

        # Prepare detections for DeepSort
        detections_for_tracker = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            detections_for_tracker.append(([x1, y1, x2 - x1, y2 - y1], score, 'player'))

        track_results = tracker.update_tracks(detections_for_tracker, frame=frame)

        frame_tracks = []
        for track in track_results:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # left, top, right, bottom
            frame_tracks.append({
                "track_id": int(track_id),
                "box": [float(x) for x in ltrb]
            })

        tracks_output.append({
            "frame_id": frame_id,
            "tracks": frame_tracks
        })

        frame_id += 1
        pbar.update(1)

    cap.release()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    # Save track results
    with open(output_json, 'w') as f:
        json.dump(tracks_output, f)

if __name__ == "__main__":
    run_tracker("football_project/videos/broadcast.mp4",
                "football_project/videos/broadcast_dets.json",
                "football_project/videos/broadcast_tracks.json")

    run_tracker("football_project/videos/tacticam.mp4",
                "football_project/videos/tacticam_dets.json",
                "football_project/videos/tacticam_tracks.json")
