---
title: Escalator Monitor
emoji: 🚦
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Detect working, stopped and idle escalators in CCTV video
---

# Escalator Monitor

Upload a clip from a fixed CCTV camera, click the four corners of the
escalator and get its state over time (working, stopped with people on it,
idle), an annotated video, a timeline and an event log.

YOLO11 finds people, dense optical flow measures whether the steps and
handrails move once people are masked out, and a hysteresis state machine
turns that into a stable status.

Source code, CLI for RTSP streams, Docker image and docs:
https://github.com/pranjal-agarwal01/Station-Facility-Monitor-Escalator-using-Computer-Vision-
