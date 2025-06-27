from ultralytics import YOLO
import cv2, pathlib, json, tqdm, os

# Define model path
model_path = pathlib.Path('football_project/model/best.pt')
model = YOLO(model_path)

def detect_players(video, out_json):
    cap = cv2.VideoCapture(video)
    frames = []
    fid = 0
    pbar = tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        scores = results[0].boxes.conf.cpu().numpy().tolist()
        frame_data = {"frame_id": fid, "boxes": boxes, "scores": scores}

        frames.append(frame_data)

        fid += 1
        pbar.update(1)

    cap.release()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    # Save detections
    with open(out_json, 'w') as f:
        json.dump(frames, f)

if __name__ == "__main__":
    detect_players('football_project/videos/broadcast.mp4', 'football_project/videos/broadcast_dets.json')
    detect_players('football_project/videos/tacticam.mp4', 'football_project/videos/tacticam_dets.json')
