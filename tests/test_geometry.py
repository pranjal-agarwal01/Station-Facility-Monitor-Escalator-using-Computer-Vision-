import numpy as np
import pytest

from escalator_monitor.geometry import Quad, build_region_masks, order_corners

TRAPEZOID = [[40, 10], [60, 10], [90, 100], [10, 100]]  # TL TR BR BL


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (2, 0, 3, 1), (3, 2, 1, 0), (1, 3, 0, 2)])
def test_corners_are_ordered_regardless_of_click_order(order):
    shuffled = [TRAPEZOID[i] for i in order]
    assert order_corners(shuffled).tolist() == TRAPEZOID


def test_parse_accepts_common_separators():
    a = Quad.parse("40,10,60,10,90,100,10,100")
    b = Quad.parse("40,10; 60,10; 90,100; 10,100")
    assert a == b
    assert a.to_list() == TRAPEZOID


def test_invalid_rois_are_rejected():
    with pytest.raises(ValueError):
        Quad.parse("1,2,3")
    with pytest.raises(ValueError):
        Quad.from_points([[0, 0], [1, 0], [1, 1], [0, 1]])  # too small


def test_bbox_margin_is_clipped_to_frame():
    q = Quad.from_points(TRAPEZOID)
    assert q.bbox() == (10, 10, 91, 101)
    x1, y1, x2, y2 = q.bbox(margin=0.5, width=100, height=105)
    assert (x1, y1) == (0, 0) and x2 == 100 and y2 == 105


def test_contains():
    q = Quad.from_points(TRAPEZOID)
    assert q.contains(50, 50)
    assert not q.contains(5, 20)


def test_region_masks_partition_the_roi():
    q = Quad.from_points([[20, 10], [80, 10], [95, 190], [5, 190]])
    window = q.bbox(0.1, 100, 200)
    shape = (window[3] - window[1], window[2] - window[0])
    m = build_region_masks(q, window, (1.0, 1.0), shape, width_frac=0.1)
    union = m.left | m.right | m.steps
    assert (union & ~m.roi.astype(bool)).sum() <= 0.02 * m.roi.sum()  # regions stay inside the ROI
    assert union.sum() >= 0.97 * m.roi.sum()  # and cover it
    cols = np.arange(shape[1])
    mean_col = {name: (getattr(m, name) * cols).sum() / getattr(m, name).sum() for name in ("left", "steps", "right")}
    assert mean_col["left"] < mean_col["steps"] < mean_col["right"]
    assert not (m.background & m.roi).any()


def test_save_and_load_roundtrip(tmp_path):
    q = Quad.from_points(TRAPEZOID)
    path = tmp_path / "roi.json"
    q.save(path)
    assert Quad.load(path) == q
