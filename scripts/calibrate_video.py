#!/usr/bin/env python3
"""Interactive per-camera calibration for analyze_video.py.

Click the six reference points in the displayed video frame. The output JSON
keeps the analysis settings from the template but replaces its homography and
visible table polygon with your clicks.
"""
import argparse
import base64
import json
from pathlib import Path
import tkinter as tk

import cv2


POINTS = (
    ("net base: upper endpoint", [-0.7625, 0.0]),
    ("net base: lower endpoint", [0.7625, 0.0]),
    ("white x=0 line: player-side table edge", [0.0, -1.37]),
    ("white x=0 line: opponent-side table edge", [0.0, 1.37]),
    ("table corner 1 (clockwise)", None),
    ("table corner 2 (clockwise)", None),
    ("table corner 3 (clockwise)", None),
    ("table corner 4 (clockwise)", None),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--template", default=str(Path(__file__).with_name("table_calibration.sample.json")))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=0, help="frame timestamp to calibrate")
    parser.add_argument("--max-width", type=int, default=1280)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, args.seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read calibration frame")
    original_height, original_width = frame.shape[:2]
    scale = min(1.0, args.max_width / original_width)
    display = cv2.resize(frame, (round(original_width * scale), round(original_height * scale)))
    ok, encoded = cv2.imencode(".png", display)
    if not ok:
        raise SystemExit("Could not encode calibration frame")

    root = tk.Tk()
    root.title("Eleven video calibration")
    instruction = tk.StringVar()
    tk.Label(root, textvariable=instruction, font=("Helvetica", 16)).pack(padx=10, pady=8)
    image = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()))
    canvas = tk.Canvas(root, width=display.shape[1], height=display.shape[0])
    canvas.pack()
    canvas.create_image(0, 0, image=image, anchor=tk.NW)
    clicked = []

    def refresh():
        if len(clicked) == len(POINTS):
            instruction.set("All points selected. Press Enter to save, Escape to cancel.")
        else:
            instruction.set(f"Click {len(clicked) + 1}/8: {POINTS[len(clicked)][0]}")

    def click(event):
        if len(clicked) >= len(POINTS):
            return
        point = [round(event.x / scale), round(event.y / scale)]
        clicked.append(point)
        canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, outline="#ff00ff", width=3)
        canvas.create_text(event.x + 12, event.y - 12, text=str(len(clicked)), fill="#ff00ff", anchor=tk.SW, font=("Helvetica", 14, "bold"))
        refresh()

    def save(_event=None):
        if len(clicked) != len(POINTS):
            return
        calibration = json.loads(Path(args.template).read_text())
        calibration["image_size"] = [original_width, original_height]
        calibration["control_points"] = [
            {"name": POINTS[index][0], "image": clicked[index], "log": POINTS[index][1]}
            for index in range(4)
        ]
        calibration["table_polygon"] = clicked[4:]
        calibration["net_line"] = clicked[:2]
        Path(args.output).write_text(json.dumps(calibration, indent=2) + "\n")
        root.destroy()

    canvas.bind("<Button-1>", click)
    root.bind("<Return>", save)
    root.bind("<Escape>", lambda _event: root.destroy())
    refresh()
    root.mainloop()


if __name__ == "__main__":
    main()
