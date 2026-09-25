// Dense optical flow through OpenCV.js. Farneback with the same parameters as
// the Python fallback (escalator_monitor/flow.py); DIS is not part of opencv.js.

export function makeFarnebackFlow(cv) {
  return (prev, curr, width, height) => {
    const a = cv.matFromArray(height, width, cv.CV_8UC1, prev);
    const b = cv.matFromArray(height, width, cv.CV_8UC1, curr);
    const flow = new cv.Mat();
    try {
      cv.calcOpticalFlowFarneback(a, b, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
      return new Float32Array(flow.data32F); // copy out of the WASM heap
    } finally {
      a.delete();
      b.delete();
      flow.delete();
    }
  };
}

/** Wait for an opencv.js module object (UMD build) to finish initialising. */
export async function readyOpenCV(cv) {
  if (cv && typeof cv.then === "function" && !cv.Mat) cv = await cv;
  if (!cv.Mat) await new Promise((resolve) => (cv.onRuntimeInitialized = resolve));
  return cv;
}
