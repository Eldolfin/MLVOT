import glob
import os
import sys

import cv2 as cv
from tracker import Tracker
from utils import load_detections, save_results


def main():
    # Configuration
    # Assumes standard MOT structure if not provided
    seq_name = "ADL-Rundle-6"
    data_root = os.path.join(os.path.dirname(__file__), "data")
    seq_path = os.path.join(data_root, seq_name)

    img_dir = os.path.join(seq_path, "img1")
    det_path = os.path.join(seq_path, "det", "det.txt")
    output_video_path = "output_tp3.avi"
    output_res_path = f"{seq_name}.txt"

    # Check if data exists
    if not os.path.exists(img_dir) or not os.path.exists(det_path):
        print(f"Error: Data not found at {seq_path}")
        print(f"Expected structure:\n{seq_path}/\n  img1/*.jpg\n  det/det.txt")
        print(
            "Please ensure the 'ADL-Rundle-6' dataset is present in a 'data' folder inside 'tp2'."
        )
        # For demonstration purposes in a CLI environment without data, we might want to exit or mock.
        # But per instructions to "Make sure it runs fine", I will allow it to fail gracefully if data is missing.
        sys.exit(1)

    # Load detections
    print(f"Loading detections from {det_path}...")
    detections = load_detections(det_path)

    # Get image list
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not images:
        print("No images found in", img_dir)
        sys.exit(1)

    # Initialize video writer
    first_img = cv.imread(images[0])
    if first_img is None:
        print("Failed to read the first image.")
        sys.exit(1)

    assert first_img is not None
    height, width, _ = first_img.shape
    fourcc = cv.VideoWriter_fourcc(*"XVID")  # type: ignore
    out = cv.VideoWriter(output_video_path, fourcc, 25.0, (width, height))

    tracker = Tracker(max_missed_frames=5, iou_threshold=0.3)

    tracking_results = []

    print(f"Processing {len(images)} frames...")

    for i, img_path in enumerate(images):
        frame = cv.imread(img_path)
        if frame is None:
            continue

        # Frame numbers in MOT usually start at 1
        frame_idx = i + 1

        current_dets = detections.get(frame_idx, [])

        # Update tracker
        tracks = tracker.update(current_dets)

        # Visualization and Result Collection
        for track in tracks:
            # Save result: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>
            x, y, w, h = track.box
            tracking_results.append((frame_idx, track.id, x, y, w, h, track.info))

            # Draw bounding box
            cv.rectangle(
                frame, (int(x), int(y)), (int(x + w), int(y + h)), track.color, 2
            )
            # Draw ID
            cv.putText(
                frame,
                str(track.id),
                (int(x), int(y) - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                track.color,
                2,
            )

        out.write(frame)

        if i % 50 == 0:
            print(f"Processed frame {i}/{len(images)}")

    out.release()

    print(f"Saving results to {output_res_path}...")
    save_results(output_res_path, tracking_results)
    print("Done.")


if __name__ == "__main__":
    main()
