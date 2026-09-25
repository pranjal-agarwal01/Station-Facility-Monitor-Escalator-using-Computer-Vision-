import logging

import pytest

from escalator_monitor.config import Config, source_name


def test_yaml_overrides_only_listed_keys(tmp_path):
    path = tmp_path / "cam.yaml"
    path.write_text("input_video: input/cam1.mp4\nhandrail_mag_gate: 0.08\nwindow_size: 60\n")
    cfg = Config.from_yaml(path)
    assert cfg.input_video == "input/cam1.mp4"
    assert cfg.handrail_mag_gate == 0.08
    assert cfg.window_size == 60
    assert cfg.steps_mag_gate == Config().steps_mag_gate


def test_unknown_keys_are_reported(tmp_path, caplog):
    path = tmp_path / "cam.yaml"
    path.write_text("handrail_mag_gat: 0.08\n")
    with caplog.at_level(logging.WARNING):
        Config.from_yaml(path)
    assert "handrail_mag_gat" in caplog.text


@pytest.mark.parametrize(
    "overrides",
    [
        {"detect_every_n_frames": 0},
        {"move_confidence_min": 1.5},
        {"exit_working_ratio": 0.6, "enter_working_ratio": 0.4},
        {"expected_direction": "left"},
        {"roi_points": [[0, 0], [1, 1]]},
    ],
)
def test_invalid_values_raise(overrides):
    with pytest.raises(ValueError):
        Config.from_dict(overrides)


def test_source_names():
    assert source_name("input/mall3.mp4") == "mall3"
    assert source_name("0") == "camera0"
    assert source_name("rtsp://user:secret@10.0.0.5:554/stream1") == "10_0_0_5_554_stream1"
    assert Config(input_video="input/stair2.mp4").resolved_roi_file().endswith("roi_stair2.json")
