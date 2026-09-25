"""Video input (files, RTSP/HTTP streams, webcams) and browser-friendly output."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

LIVE_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def parse_source(source: str | int) -> str | int:
    text = str(source).strip()
    return int(text) if text.isdigit() else text


def is_live(source: str | int) -> bool:
    src = parse_source(source)
    return isinstance(src, int) or src.lower().startswith(LIVE_PREFIXES)


@dataclass
class Frame:
    index: int  # 1-based position in the source
    time: float  # seconds since the start (video time for files, wall time for streams)
    image: np.ndarray


class _LatestFrameReader:
    """Reads a live stream on a background thread and keeps only the newest frame,
    so slow processing never builds up latency."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._ended = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            ok, frame = self.cap.read()
            with self._cond:
                if not ok:
                    self._ended = True
                    self._cond.notify_all()
                    return
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()

    def read(self, last_seq: int, timeout: float = 10.0) -> tuple[int, np.ndarray | None]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq > last_seq or self._ended, timeout=timeout)
            if self._seq > last_seq:
                return self._seq, self._frame
            return last_seq, None


class VideoSource:
    """Iterates frames from a file, stream or camera, optionally downscaled."""

    def __init__(self, source: str | int, max_width: int = 0, stride: int = 1):
        self.source = parse_source(source)
        self.live = is_live(source)
        if not self.live and not Path(str(self.source)).exists():
            raise FileNotFoundError(f"Input video not found: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise OSError(f"Could not open video source: {self.source}")
        self.stride = 1 if self.live else max(1, stride)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not 1.0 <= self.fps <= 240.0:
            self.fps = 25.0
        self.total_frames = 0 if self.live else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale = max_width / src_w if max_width and src_w > max_width else 1.0
        self.width, self.height = round(src_w * self.scale), round(src_h * self.scale)
        self._reader = _LatestFrameReader(self.cap) if self.live else None

    @property
    def output_fps(self) -> float:
        return self.fps / self.stride

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.scale != 1.0:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def __iter__(self) -> Iterator[Frame]:
        if self._reader is not None:
            yield from self._iter_live()
            return
        index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            index += 1
            yield Frame(index, (index - 1) / self.fps, self._resize(frame))
            for _ in range(self.stride - 1):
                if not self.cap.grab():
                    return
                index += 1

    def _iter_live(self) -> Iterator[Frame]:
        t0 = time.monotonic()
        seq, index = 0, 0
        while True:
            seq, frame = self._reader.read(seq)
            if frame is None:
                log.warning("Stream ended or timed out: %s", self.source)
                return
            index += 1
            yield Frame(index, time.monotonic() - t0, self._resize(frame))

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class VideoWriter:
    """H.264 MP4 through the ffmpeg binary bundled with imageio-ffmpeg (plays in
    browsers); falls back to OpenCV's mp4v encoder if that is not installed."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        self.path = str(path)
        self.size = size
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._gen = None
        self._cv = None
        try:
            import imageio_ffmpeg

            self._gen = imageio_ffmpeg.write_frames(
                self.path,
                size,
                pix_fmt_in="bgr24",
                fps=fps,
                codec="libx264",
                quality=None,
                macro_block_size=2,
                ffmpeg_log_level="error",
                output_params=["-crf", "23", "-preset", "veryfast", "-movflags", "+faststart"],
            )
            self._gen.send(None)
            self.codec = "h264"
        except Exception as exc:  # ImportError or a missing ffmpeg binary
            if self._gen is not None:
                self._gen.close()
                self._gen = None
            log.info("H.264 writer unavailable (%s); using OpenCV mp4v", exc)
            self._cv = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not self._cv.isOpened():
                raise OSError(f"Could not open video writer for {self.path}") from exc
            self.codec = "mp4v"

    def write(self, frame: np.ndarray) -> None:
        if self._gen is not None:
            self._gen.send(np.ascontiguousarray(frame))
        else:
            self._cv.write(frame)

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
            self._gen = None
        if self._cv is not None:
            self._cv.release()
            self._cv = None
