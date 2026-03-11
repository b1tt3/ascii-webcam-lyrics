/*  Screen → ASCII with Hand Tracking + Lyrics (p5.js)
    Requires in index.html:
      - p5.min.js
      - @tensorflow/tfjs-core, tfjs-backend-webgl, tfjs-converter
      - @tensorflow-models/hand-pose-detection
*/

let videoElt, handpose, backgroundBuffer, lyricsInput, submitButton;
let startBtn, stopBtn;
let screenStream = null;

let predictions = [];
let isVideoReady = false;

let lyrics = ["WHAT", "TO", "DO"];
let currentLyricIndex = 0;
let frameCounter = 0;

const LYRIC_CHANGE_INTERVAL = 30;
const ASCII_CHARS = ['@', '#', 'B', 'I', 'T', '*', 'E', ';', ':', ',', '.'];
const CELL_W = 10, CELL_H = 18;
const DIFF_THRESHOLD = 50;
const MAX_HANDS = 4;

let isInputActive = false;

function setup() {
  // Force pixel density 1 so pixel-index math is straightforward
  pixelDensity(1);

  createCanvas(640, 480);

  // Create a raw <video> element — avoids p5's createVideo([]) quirks
  videoElt = document.createElement('video');
  videoElt.muted = true;
  videoElt.playsInline = true;
  videoElt.width = width;
  videoElt.height = height;
  videoElt.style.display = 'none';
  document.body.appendChild(videoElt);

  textFont('Courier');
  textAlign(CENTER, CENTER);

  // Also force density 1 on the offscreen buffer
  backgroundBuffer = createGraphics(width, height);
  backgroundBuffer.pixelDensity(1);

  createControls();
  initializeHandPose();
}

function createControls() {
  startBtn = createButton('Start Screen (S)');
  startBtn.position(10, 10);
  startBtn.mousePressed(startScreenFeed);

  stopBtn = createButton('Stop');
  stopBtn.position(130, 10);
  stopBtn.mousePressed(stopScreenFeed);

  // Lyrics editor
  lyricsInput = createElement('textarea');
  lyricsInput.position(10, 50);
  lyricsInput.size(width - 20, 120);
  lyricsInput.style('font-size', '14px');
  lyricsInput.style('display', 'none');
  lyricsInput.attribute('placeholder', 'Enter lyrics, one per line');

  submitButton = createButton('Submit Lyrics');
  submitButton.position(10, 180);
  submitButton.mousePressed(submitLyrics);
  submitButton.style('display', 'none');
}

async function initializeHandPose() {
  // Use WebGL backend for speed
  if (window.tf && tf.getBackend() !== 'webgl') {
    try { await tf.setBackend('webgl'); await tf.ready(); } catch(e) {}
  }
  handpose = await handPoseDetection.createDetector(
    handPoseDetection.SupportedModels.MediaPipeHands,
    { runtime: "tfjs", modelType: "full", maxHands: MAX_HANDS }
  );
  detectHandsLoop();
}

function detectHandsLoop() {
  const tick = async () => {
    if (isVideoReady && videoElt.readyState >= 2) {
      try {
        predictions = await handpose.estimateHands(videoElt);
      } catch (_) { /* occasionally throws; ignore */ }
    }
    setTimeout(tick, 100); // ~10Hz is plenty
  };
  tick();
}

async function startScreenFeed() {
  try {
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: 30 }, audio: false
    });
    screenStream = stream;

    videoElt.srcObject = screenStream;
    await videoElt.play();
    isVideoReady = true;

    // If user stops sharing from browser UI
    const track = screenStream.getVideoTracks()[0];
    track.addEventListener('ended', () => {
      isVideoReady = false;
      videoElt.srcObject = null;
    });
  } catch (e) {
    console.warn('Screen capture denied/failed:');
    console.warn('  name:', e.name);
    console.warn('  message:', e.message);
    console.warn('  protocol:', window.location.protocol);
    console.warn('  inIframe:', window.self !== window.top);
    if (e.name === 'NotAllowedError') {
      console.warn('  → User denied the prompt, or browser blocked it (iframe / policy).');
    } else if (e.name === 'NotSupportedError') {
      console.warn('  → getDisplayMedia not supported in this context.');
    }
  }
}

function stopScreenFeed() {
  if (screenStream) {
    screenStream.getTracks().forEach(t => t.stop());
    screenStream = null;
  }
  isVideoReady = false;
  videoElt.srcObject = null;
}

function draw() {
  frameCounter++;
  background(0);

  if (!isVideoReady || !videoElt.srcObject) {
    fill(200);
    textSize(14);
    text('Press "Start Screen (S)" to begin screen capture.', width/2, height/2);
    return;
  }

  drawSubtleAsciiVideo();

  // Draw hands + fingertip lyrics
  stroke(255, 255, 0);
  strokeWeight(3);
  noFill();

  for (const hand of predictions) {
    if (!hand?.keypoints) continue;

    // keypoints
    for (const kp of hand.keypoints) point(kp.x, kp.y);

    // lyrics at fingertips
    const fingerTips = [4, 8, 12, 16, 20];
    noStroke();
    fill(0, 255, 0);
    textSize(16);
    fingerTips.forEach((idx, i) => {
      const tip = hand.keypoints[idx];
      if (tip) {
        const lidx = (currentLyricIndex + i) % lyrics.length;
        text(lyrics[lidx], tip.x, tip.y - 15);
      }
    });
  }

  if (frameCounter % LYRIC_CHANGE_INTERVAL === 0) {
    currentLyricIndex = (currentLyricIndex + 1) % lyrics.length;
  }
}

function drawSubtleAsciiVideo() {
  // Draw the video into a temp p5.Graphics so loadPixels() works reliably
  // (p5's video.loadPixels() can return empty with srcObject-fed streams)
  let pg = createGraphics(width, height);
  pg.pixelDensity(1);
  pg.drawingContext.drawImage(videoElt, 0, 0, width, height);
  pg.loadPixels();

  backgroundBuffer.loadPixels();

  for (let y = 0; y < height; y += CELL_H) {
    for (let x = 0; x < width; x += CELL_W) {
      const idx = (y * width + x) * 4;
      const r = pg.pixels[idx]     || 0;
      const g = pg.pixels[idx + 1] || 0;
      const b = pg.pixels[idx + 2] || 0;

      const br = backgroundBuffer.pixels[idx]     || 0;
      const bg = backgroundBuffer.pixels[idx + 1] || 0;
      const bb = backgroundBuffer.pixels[idx + 2] || 0;

      const diff = Math.abs(r - br) + Math.abs(g - bg) + Math.abs(b - bb);

      if (diff > DIFF_THRESHOLD) {
        const rr = constrain(r * 1.2, 0, 255);
        const gg = constrain(g * 0.8, 0, 255);
        const bb2 = constrain(b * 1.1, 0, 255);
        const avg = (rr + gg + bb2) / 3;
        const ci = floor(map(avg, 0, 255, ASCII_CHARS.length - 1, 0));

        noStroke();
        fill(rr, gg, bb2);
        textSize(16);
        text(ASCII_CHARS[ci], x + CELL_W/2, y + CELL_H/2);
      }

      // EMA into background buffer
      backgroundBuffer.pixels[idx]     = br * 0.95 + r * 0.05;
      backgroundBuffer.pixels[idx + 1] = bg * 0.95 + g * 0.05;
      backgroundBuffer.pixels[idx + 2] = bb * 0.95 + b * 0.05;
    }
  }

  backgroundBuffer.updatePixels();
  pg.remove(); // clean up temp graphics each frame
}

function keyPressed() {
  // Don't intercept keys while typing lyrics
  if (isInputActive) return;

  if (key === 'L' || key === 'l') {
    toggleLyricsInput();
  } else if (key === 'S' || key === 's') {
    startScreenFeed();
  }
}

function toggleLyricsInput() {
  isInputActive = !isInputActive;
  lyricsInput.style('display', isInputActive ? 'block' : 'none');
  submitButton.style('display', isInputActive ? 'block' : 'none');
  if (!isInputActive) updateLyrics();
}

function submitLyrics() {
  isInputActive = false;
  lyricsInput.style('display', 'none');
  submitButton.style('display', 'none');
  updateLyrics();
}

function updateLyrics() {
  const lines = lyricsInput.value().split('\n').map(s => s.trim()).filter(Boolean);
  if (lines.length) {
    lyrics = lines;
    currentLyricIndex = 0;
  }
}
