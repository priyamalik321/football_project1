import json
import cv2
import numpy as np
from collections import defaultdict
import os

def load_tracks(track_file):
    with open(track_file, 'r') as f:
        return json.load(f)

def extract_histogram(frame, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((180,))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def build_feature_dict(video_path, tracks):
    cap = cv2.VideoCapture(video_path)
    features = defaultdict(list)

    for t in tracks:
        frame_id = t['frame_id']
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        for obj in t['tracks']:
            tid = obj['track_id']
            box = obj['box']
            hist = extract_histogram(frame, box)
            features[tid].append(hist)

    cap.release()
    # Average histograms per player
    averaged = {tid: np.mean(hists, axis=0) for tid, hists in features.items()}
    return averaged

def match_players(broadcast_feats, tacticam_feats):
    mapping = {}
    used_broadcast_ids = set()

    for t_id, t_feat in tacticam_feats.items():
        best_match = None
        best_score = float('inf')

        for b_id, b_feat in broadcast_feats.items():
            if b_id in used_broadcast_ids:
                continue
            dist = np.linalg.norm(t_feat - b_feat)
            if dist < best_score:
                best_score = dist
                best_match = b_id

        if best_match is not None:
            mapping[t_id] = best_match
            used_broadcast_ids.add(best_match)

    return mapping

def save_mapping(mapping, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(mapping, f, indent=4)

if __name__ == "__main__":
    broadcast_tracks = load_tracks("football_project/videos/broadcast_tracks.json")
    tacticam_tracks = load_tracks("football_project/videos/tacticam_tracks.json")

    broadcast_feats = build_feature_dict("football_project/videos/broadcast.mp4", broadcast_tracks)
    tacticam_feats = build_feature_dict("football_project/videos/tacticam.mp4", tacticam_tracks)

    player_mapping = match_players(broadcast_feats, tacticam_feats)

    save_mapping(player_mapping, "football_project/videos/tacticam_to_broadcast_id_map.json")

    print("✅ Mapping complete. Output saved to tacticam_to_broadcast_id_map.json")
