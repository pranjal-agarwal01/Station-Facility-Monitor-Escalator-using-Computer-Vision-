import pytest

from escalator_monitor.tracking import Detection, IoUTracker, iou


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_tracks_keep_ids_and_coast_between_detections():
    t = IoUTracker(iou_threshold=0.3, max_misses=2)
    first = t.update([Detection(0, 0, 10, 20), Detection(50, 0, 60, 20)])
    ids = {tr.id for tr in first}
    moved = t.update([Detection(2, 0, 12, 20), Detection(51, 0, 61, 20)])
    assert {tr.id for tr in moved} == ids
    assert len(t.update(None)) == 2  # no detector run: keep boxes


def test_tracks_expire_after_missed_detections():
    t = IoUTracker(max_misses=2)
    t.update([Detection(0, 0, 10, 20)])
    assert len(t.update([])) == 1
    assert len(t.update([])) == 1
    assert t.update([]) == []


def test_best_overlap_wins():
    t = IoUTracker()
    (a,) = t.update([Detection(0, 0, 10, 10)])
    tracks = t.update([Detection(30, 0, 40, 10), Detection(1, 0, 11, 10)])
    same = [tr for tr in tracks if tr.id == a.id]
    assert same and same[0].det.x1 == 1
    assert len(tracks) == 2
