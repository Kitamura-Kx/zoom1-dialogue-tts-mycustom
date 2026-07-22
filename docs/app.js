const players = [...document.querySelectorAll("[data-player]")];
let activeAudio = null;

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  return `${String(Math.floor(safe / 60)).padStart(2, "0")}:${String(Math.floor(safe % 60)).padStart(2, "0")}`;
}

function drawChannel(canvas, data, color) {
  const context = canvas.getContext("2d");
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);
  context.fillStyle = color;
  const step = Math.max(1, Math.floor(data.length / width));
  const center = height / 2;
  for (let x = 0; x < width; x += 2) {
    let peak = 0;
    const start = x * step;
    for (let index = start; index < Math.min(start + step * 2, data.length); index++) {
      peak = Math.max(peak, Math.abs(data[index]));
    }
    const magnitude = Math.max(1, peak * height * 0.92);
    context.fillRect(x, center - magnitude / 2, 1.5, magnitude);
  }
}

async function loadWaveforms(root, audio) {
  try {
    const response = await fetch(audio.currentSrc || audio.src);
    const arrayBuffer = await response.arrayBuffer();
    const audioContext = new AudioContext();
    const buffer = await audioContext.decodeAudioData(arrayBuffer);
    root.querySelectorAll("[data-wave-channel]").forEach((canvas) => {
      const channel = Number(canvas.dataset.waveChannel);
      drawChannel(canvas, buffer.getChannelData(Math.min(channel, buffer.numberOfChannels - 1)), channel === 0 ? "#087f73" : "#e4572e");
    });
    root.querySelectorAll("[data-wave-mix]").forEach((canvas) => {
      const left = buffer.getChannelData(0);
      const right = buffer.getChannelData(Math.min(1, buffer.numberOfChannels - 1));
      const mixed = new Float32Array(left.length);
      for (let i = 0; i < left.length; i++) mixed[i] = (left[i] + right[i]) * 0.5;
      drawChannel(canvas, mixed, root.dataset.player.includes("vap") ? "#e4572e" : "#087f73");
    });
    audioContext.close();
  } catch (error) {
    root.querySelectorAll("canvas").forEach((canvas) => {
      const context = canvas.getContext("2d");
      context.fillStyle = "#ccd2cd";
      context.fillRect(0, canvas.height / 2, canvas.width, 1);
    });
  }
}

function updatePlayer(root, audio) {
  const progress = audio.duration ? audio.currentTime / audio.duration : 0;
  root.querySelectorAll(".playhead").forEach((head) => { head.style.left = `${progress * 100}%`; });
  root.querySelectorAll(".elapsed").forEach((label) => { label.textContent = formatTime(audio.currentTime); });
  const scrubber = root.querySelector(".scrubber");
  if (scrubber && document.activeElement !== scrubber) scrubber.value = String(Math.round(progress * 1000));
}

players.forEach((root) => {
  const audio = root.querySelector("audio");
  const button = root.querySelector(".play-button");
  const scrubber = root.querySelector(".scrubber");
  let animationFrame = null;

  const stopProgressAnimation = () => {
    if (animationFrame !== null) cancelAnimationFrame(animationFrame);
    animationFrame = null;
    updatePlayer(root, audio);
  };

  const animateProgress = () => {
    updatePlayer(root, audio);
    if (!audio.paused && !audio.ended) {
      animationFrame = requestAnimationFrame(animateProgress);
    } else {
      animationFrame = null;
    }
  };

  loadWaveforms(root, audio);

  button.addEventListener("click", async () => {
    if (activeAudio && activeAudio !== audio) activeAudio.pause();
    if (audio.paused) {
      activeAudio = audio;
      await audio.play();
    } else {
      audio.pause();
    }
  });
  audio.addEventListener("play", () => {
    root.classList.add("is-playing");
    if (animationFrame === null) animationFrame = requestAnimationFrame(animateProgress);
  });
  audio.addEventListener("pause", () => {
    root.classList.remove("is-playing");
    stopProgressAnimation();
  });
  audio.addEventListener("ended", () => {
    root.classList.remove("is-playing");
    stopProgressAnimation();
  });
  audio.addEventListener("timeupdate", () => updatePlayer(root, audio));
  if (scrubber) {
    scrubber.addEventListener("input", () => {
      root.classList.add("is-seeking");
      if (audio.duration) audio.currentTime = Number(scrubber.value) / 1000 * audio.duration;
      updatePlayer(root, audio);
    });
    scrubber.addEventListener("change", () => root.classList.remove("is-seeking"));
  }
});

document.querySelectorAll(".copy-button").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    await navigator.clipboard.writeText(target.innerText.replace(/^\$ /gm, ""));
    button.textContent = "Copied";
    setTimeout(() => { button.textContent = "Copy"; }, 1400);
  });
});
