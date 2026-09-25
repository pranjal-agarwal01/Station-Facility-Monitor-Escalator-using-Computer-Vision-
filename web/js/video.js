// Frame-accurate access to a video file in the browser: seek-based stepping
// (deterministic, works for any codec the browser can play) and an MP4 reader
// that recovers the frame rate, which HTMLVideoElement does not expose.

function* boxes(view, start, end) {
  let pos = start;
  while (pos + 8 <= end) {
    let size = view.getUint32(pos);
    const type = String.fromCharCode(
      view.getUint8(pos + 4),
      view.getUint8(pos + 5),
      view.getUint8(pos + 6),
      view.getUint8(pos + 7),
    );
    let header = 8;
    if (size === 1) {
      size = Number(view.getBigUint64(pos + 8));
      header = 16;
    } else if (size === 0) {
      size = end - pos;
    }
    if (size < header) return;
    yield { type, start: pos + header, end: Math.min(end, pos + size) };
    pos += size;
  }
}

function findChild(view, parent, type) {
  for (const b of boxes(view, parent.start, parent.end)) if (b.type === type) return b;
  return null;
}

function videoTrackFps(view, moov) {
  for (const trak of boxes(view, moov.start, moov.end)) {
    if (trak.type !== "trak") continue;
    const mdia = findChild(view, trak, "mdia");
    const hdlr = mdia && findChild(view, mdia, "hdlr");
    if (!hdlr) continue;
    const handler = String.fromCharCode(...new Uint8Array(view.buffer, view.byteOffset + hdlr.start + 8, 4));
    if (handler !== "vide") continue;
    const mdhd = findChild(view, mdia, "mdhd");
    const version = view.getUint8(mdhd.start);
    const timescale = view.getUint32(mdhd.start + (version === 1 ? 20 : 12));
    let stbl = findChild(view, mdia, "minf");
    stbl = stbl && findChild(view, stbl, "stbl");
    const stts = stbl && findChild(view, stbl, "stts");
    if (!stts || !timescale) return null;
    const entries = view.getUint32(stts.start + 4);
    let frames = 0;
    let ticks = 0;
    for (let i = 0; i < entries; i++) {
      const count = view.getUint32(stts.start + 8 + i * 8);
      const delta = view.getUint32(stts.start + 12 + i * 8);
      frames += count;
      ticks += count * delta;
    }
    return frames && ticks ? (frames * timescale) / ticks : null;
  }
  return null;
}

/** Frame rate of an MP4/MOV file, or null if it can't be read (e.g. fragmented MP4). */
export async function probeFps(blob) {
  try {
    let pos = 0;
    while (pos + 16 <= blob.size) {
      const head = new DataView(await blob.slice(pos, pos + 16).arrayBuffer());
      let size = head.getUint32(0);
      const type = String.fromCharCode(head.getUint8(4), head.getUint8(5), head.getUint8(6), head.getUint8(7));
      if (size === 1) size = Number(head.getBigUint64(8));
      else if (size === 0) size = blob.size - pos;
      if (size < 8) return null;
      if (type === "moov") {
        const view = new DataView(await blob.slice(pos, pos + size).arrayBuffer());
        const moov = [...boxes(view, 0, view.byteLength)][0];
        const fps = videoTrackFps(view, moov);
        return fps && fps > 1 && fps < 241 ? fps : null;
      }
      pos += size;
    }
  } catch (err) {
    console.warn("Could not read the frame rate", err);
  }
  return null;
}

/** Load a video URL into a <video> element; resolves once the first frame is decodable. */
export function loadVideo(video, url) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener("loadeddata", ok);
      video.removeEventListener("error", fail);
    };
    const ok = () => {
      cleanup();
      resolve(video);
    };
    const fail = () => {
      cleanup();
      reject(new Error("This browser can't decode that video. Try an MP4 (H.264) file."));
    };
    video.addEventListener("loadeddata", ok);
    video.addEventListener("error", fail);
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.src = url;
    video.load();
  });
}

/** Seek and wait until the frame at `time` can be drawn. */
export function seek(video, time) {
  return new Promise((resolve) => {
    const done = () => {
      clearTimeout(timer);
      video.removeEventListener("seeked", done);
      resolve();
    };
    const timer = setTimeout(done, 4000); // never hang on a stubborn decoder
    video.addEventListener("seeked", done);
    video.currentTime = time;
  });
}

const COMMON_RATES = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60];

/** Measure the frame rate by playing the (muted) video briefly. Null if unsupported. */
export async function estimateFps(video, frames = 12) {
  if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) return null;
  const times = [];
  await new Promise((resolve) => {
    const timer = setTimeout(resolve, 3000);
    const onFrame = (_now, meta) => {
      times.push(meta.mediaTime);
      if (times.length >= frames) {
        clearTimeout(timer);
        resolve();
      } else {
        video.requestVideoFrameCallback(onFrame);
      }
    };
    video.requestVideoFrameCallback(onFrame);
    video.play().catch(() => {
      clearTimeout(timer);
      resolve();
    });
  });
  video.pause();
  const deltas = times
    .slice(1)
    .map((t, i) => t - times[i])
    .filter((d) => d > 1e-4)
    .sort((a, b) => a - b);
  if (deltas.length < 3) return null;
  const fps = 1 / deltas[deltas.length >> 1];
  const snap = COMMON_RATES.find((r) => Math.abs(r - fps) / r < 0.03);
  return snap ?? (fps > 1 && fps < 241 ? fps : null);
}
