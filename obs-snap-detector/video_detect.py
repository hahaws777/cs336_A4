"""
video_detect.py — Run person + head-zone detection on a video file.

This is an offline video annotator. It reads an input video, runs the existing
Detector, optionally smooths boxes with ByteTracker, and writes an annotated
output video with:
  - person/body boxes
  - estimated head-zone boxes
  - snap points near the head zone

Examples:
    python video_detect.py --input input.mp4
    python video_detect.py --input input.mp4 --output annotated.mp4 --confidence 0.25
    python video_detect.py --input input.mp4 --no-tracker --max-frames 300
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        return (0, 255, 0)
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


def _default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_detected.mp4")


def _open_writer(output: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    import cv2

    output.parent.mkdir(parents=True, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), codec, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output}")
    return writer


def _draw_detection(frame, det, is_primary: bool) -> None:
    import cv2

    from config import BOX_COLOR, HEAD_ZONE_COLOR, PRIMARY_TARGET_COLOR, SNAP_POINT_COLOR

    box_color = _hex_to_bgr(PRIMARY_TARGET_COLOR if is_primary else BOX_COLOR)
    head_color = _hex_to_bgr(HEAD_ZONE_COLOR)
    point_color = _hex_to_bgr(SNAP_POINT_COLOR)

    cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), box_color, 2 if not is_primary else 3)

    hx1, hy1, hx2, hy2 = det.head_box
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), head_color, 2)

    sx, sy = det.snap_point
    cv2.circle(frame, (sx, sy), 4 if is_primary else 3, point_color, -1)

    label = f"ID {det.track_id} " if det.track_id >= 0 else ""
    label += f"{det.confidence:.0%}"
    if is_primary:
        label += f"  {det.distance_to_center:.0f}px"

    y_text = max(det.y1 - 8, 18)
    cv2.putText(frame, label, (det.x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate a video with person boxes and estimated head zones.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output", "-o", type=Path, help="Output video path. Defaults to <input>_detected.mp4.")
    parser.add_argument("--confidence", type=float, help="Override config.CONFIDENCE_THRESHOLD.")
    parser.add_argument("--head-zone-ratio", type=float, help="Override head-zone height ratio, e.g. 0.20.")
    parser.add_argument("--no-tracker", action="store_true", help="Draw raw detections without ByteTracker smoothing.")
    parser.add_argument("--no-hand-filter", action="store_true", help="Disable lower-screen hand/weapon filtering for generic videos.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames; 0 means full video.")
    parser.add_argument("--start-frame", type=int, default=0, help="Start reading at this frame index.")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start reading at this timestamp in seconds.")
    parser.add_argument("--show", action="store_true", help="Preview while processing. Press Q to stop.")
    parser.add_argument("--every-n", type=int, default=1, help="Run detection every N frames and reuse last result between frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.every_n < 1:
        raise ValueError("--every-n must be >= 1")

    input_path = args.input
    output_path = args.output or _default_output(input_path)

    import cv2

    # Apply lightweight config overrides before importing detector constants.
    import config

    if args.confidence is not None:
        config.CONFIDENCE_THRESHOLD = args.confidence
    if args.head_zone_ratio is not None:
        config.HEAD_ZONE_RATIO = args.head_zone_ratio

    import detector as detector_module
    from detector import Detector
    from tracker import ByteTracker

    if args.confidence is not None:
        detector_module.CONFIDENCE_THRESHOLD = args.confidence
    if args.head_zone_ratio is not None:
        detector_module.HEAD_ZONE_RATIO = args.head_zone_ratio

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(args.start_frame, int(args.start_time * fps))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = _open_writer(output_path, fps, width, height)
    detector = Detector(screen_size=(width, height))
    tracker = None if args.no_tracker else ByteTracker()

    if args.no_hand_filter:
        detector._is_hand = lambda *_: False

    frame_index = start_frame
    processed = 0
    written = 0
    last_detections = []
    started = time.perf_counter()

    print(f"[Video] input:  {input_path}")
    print(f"[Video] output: {output_path}")
    print(f"[Video] size: {width}x{height}, fps={fps:.2f}, frames={total or 'unknown'}")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if processed % args.every_n == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                raw = detector.detect(frame_rgb)
                last_detections = tracker.update(raw, width, height) if tracker else raw

            detections = list(last_detections)
            for index, det in enumerate(detections):
                _draw_detection(frame_bgr, det, index == 0)

            cv2.putText(
                frame_bgr,
                f"frame {frame_index + 1}  targets {len(detections)}",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame_bgr)
            written += 1

            if args.show:
                cv2.imshow("video_detect", frame_bgr)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break

            frame_index += 1
            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break

            if processed % 100 == 0:
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(f"[Video] processed {processed} frames ({processed / elapsed:.1f} fps)")
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - started, 1e-6)
    print(f"[Video] done: wrote {written} frames in {elapsed:.1f}s -> {output_path}")


if __name__ == "__main__":
    main()
