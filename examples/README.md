# Example clips for the web demo

Any `*.mp4` placed here shows up under **Examples** in the web app (`app.py`),
next to the two synthetic clips it renders at start-up.

Add a matching `<name>.roi.json` so the escalator region is pre-filled:

```json
{"points": [[812, 140], [1105, 140], [1290, 1040], [640, 1040]]}
```

Points are the escalator's top-left, top-right, bottom-right and bottom-left
corners in source-video pixels. `escalator-monitor run` saves exactly this
file (`output/roi_<name>.json`) after you click the corners once.

Keep clips short (10-30 s) and small (< 20 MB), and only commit footage you
have the right to share: blur faces if they are recognisable.
