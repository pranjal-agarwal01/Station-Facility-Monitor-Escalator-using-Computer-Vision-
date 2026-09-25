"""Person detections and a lightweight IoU tracker that coasts between detector runs."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import Box


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float = 1.0

    @property
    def box(self) -> Box:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def foot(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, float(self.y2)


@dataclass
class Track:
    id: int
    det: Detection
    misses: int = 0
    hits: int = 1


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    """Greedy IoU association.

    The detector only runs every few frames; in between, ``update(None)`` keeps
    the last known boxes so people stay masked out of the flow. A track is
    dropped after ``max_misses`` consecutive detector runs without a match.
    """

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 2):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.tracks: dict[int, Track] = {}
        self._next_id = 0

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, detections: list[Detection] | None) -> list[Track]:
        if detections is None:
            return list(self.tracks.values())

        pairs = sorted(
            (
                (iou(det.box, track.det.box), di, tid)
                for di, det in enumerate(detections)
                for tid, track in self.tracks.items()
            ),
            reverse=True,
        )
        used_dets: set[int] = set()
        used_tracks: set[int] = set()
        for score, di, tid in pairs:
            if score < self.iou_threshold:
                break
            if di in used_dets or tid in used_tracks:
                continue
            track = self.tracks[tid]
            track.det, track.misses = detections[di], 0
            track.hits += 1
            used_dets.add(di)
            used_tracks.add(tid)

        for tid in list(self.tracks):
            if tid not in used_tracks:
                self.tracks[tid].misses += 1
                if self.tracks[tid].misses > self.max_misses:
                    del self.tracks[tid]

        for di, det in enumerate(detections):
            if di not in used_dets:
                self.tracks[self._next_id] = Track(self._next_id, det)
                self._next_id += 1

        return list(self.tracks.values())
