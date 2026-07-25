export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function clamp(value, minimum = 0, maximum = 1) {
  return Math.min(maximum, Math.max(minimum, value));
}

export function lerp(start, end, progress) {
  return start + (end - start) * progress;
}

export function scale(value, domain, range) {
  return range[0]
    + ((value - domain[0]) / (domain[1] - domain[0]))
    * (range[1] - range[0]);
}

export function resizeCanvas(canvas, context, draw) {
  const rect = canvas.getBoundingClientRect();
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * pixelRatio));
  const height = Math.max(1, Math.round(rect.height * pixelRatio));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  draw();
}

export function observeCanvas(canvas, context, draw) {
  const resize = () => resizeCanvas(canvas, context, draw);
  const observer = new ResizeObserver(resize);
  observer.observe(canvas);
  requestAnimationFrame(resize);
  return { redraw: resize, disconnect: () => observer.disconnect() };
}

export function createMarkerController({
  stage,
  track,
  steps,
  rangeDvh,
  reducedMotion,
  onProgress,
}) {
  const markers = [];

  for (let index = 0; index < steps; index += 1) {
    const progress = index / (steps - 1);
    const marker = document.createElement("span");
    marker.className = "scroll-marker";
    marker.dataset.progress = progress.toFixed(5);
    marker.style.top = `${50 + progress * rangeDvh}dvh`;
    track.append(marker);
    markers.push(marker);
  }

  const endMarker = document.createElement("span");
  endMarker.className = "scroll-marker";
  endMarker.style.top = "calc(100% - 1px)";
  track.append(endMarker);

  if (reducedMotion.matches || !("IntersectionObserver" in window)) {
    onProgress(1);
    return { disconnect() {} };
  }

  const activeMarkers = new Set();
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) activeMarkers.add(entry.target);
      else activeMarkers.delete(entry.target);
    }
    if (activeMarkers.size === 0) return;

    const viewportCenter = window.innerHeight / 2;
    const closest = [...activeMarkers].sort((left, right) => (
      Math.abs(left.getBoundingClientRect().top - viewportCenter)
      - Math.abs(right.getBoundingClientRect().top - viewportCenter)
    ))[0];
    onProgress(Number(closest.dataset.progress));
  }, {
    root: null,
    rootMargin: "-45% 0px -45% 0px",
    threshold: 0,
  });

  for (const marker of markers) observer.observe(marker);

  const completionObserver = new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting) onProgress(1);
  }, {
    root: null,
    // The final marker completes as it crosses the viewport center. A broad
    // top-half root also catches large wheel, touchpad, and Page Down jumps.
    rootMargin: "0px 0px -50% 0px",
    threshold: 0,
  });
  completionObserver.observe(markers.at(-1));

  const endObserver = new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting) onProgress(1);
  });
  endObserver.observe(endMarker);

  stage.dataset.scrollController = "intersection-markers";

  return {
    disconnect() {
      observer.disconnect();
      completionObserver.disconnect();
      endObserver.disconnect();
    },
  };
}
