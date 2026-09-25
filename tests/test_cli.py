import json

from escalator_monitor.cli import main


def test_synth_run_evaluate(tmp_path, capsys):
    video = tmp_path / "demo.mp4"
    assert main(["synth", str(video), "--segments", "moving:3:2,stopped:4:3", "--width", "480", "--height", "360"]) == 0
    roi = str(tmp_path / "demo.roi.json")
    out = tmp_path / "out"
    code = main(
        [
            "run",
            "--input", str(video),
            "--roi", roi,
            "--detector", f"replay:{tmp_path / 'demo.boxes.json'}",
            "--headless",
            "--output-dir", str(out),
        ]
    )  # fmt: skip
    assert code == 0
    assert (out / "result.mp4").exists() and (out / "report.json").exists()
    assert (
        main(["evaluate", str(out / "timeline.csv"), str(tmp_path / "demo.gt.csv"), "--json", str(out / "m.json")]) == 0
    )
    assert json.loads((out / "m.json").read_text())["accuracy"] > 0.7
    assert "Accuracy" in capsys.readouterr().out


def test_run_is_the_default_command(tmp_path):
    # Old invocation style: flags without a sub-command. A missing file is reported, not raised.
    assert main(["--input", str(tmp_path / "missing.mp4"), "--headless", "--detector", "none"]) == 1


def test_bad_roi_is_reported(tmp_path):
    assert main(["run", "--input", "x.mp4", "--roi", "1,2,3", "--headless", "--detector", "none"]) == 1
