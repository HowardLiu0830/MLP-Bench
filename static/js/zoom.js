// Zoom and pan for figures marked with [data-zoom]. The image stays inside its card:
// buttons, Ctrl/Cmd + wheel, trackpad or touch pinch, and double-click zoom; drag pans.
// Plain wheel scrolling only pans when zoomed, and hands off to the page at the edges.
(function () {
  const MIN = 1, MAX = 6, STEP = 1.5;

  document.querySelectorAll('[data-zoom]').forEach((root) => {
    const viewport = root.querySelector('.zoom-viewport');
    const img = viewport.querySelector('img');
    const level = root.querySelector('.zoom-level');
    const btnIn = root.querySelector('[data-act="in"]');
    const btnOut = root.querySelector('[data-act="out"]');
    const btnReset = root.querySelector('[data-act="reset"]');
    let s = 1, x = 0, y = 0;
    let hdLoaded = false;

    const size = () => ({ w: viewport.clientWidth, h: viewport.clientHeight });

    function clamp() {
      const { w, h } = size();
      x = Math.min(0, Math.max(w - w * s, x));
      y = Math.min(0, Math.max(h - h * s, y));
    }

    function apply(animate) {
      clamp();
      root.classList.toggle('is-animating', !!animate);
      img.style.transform = `translate(${x}px, ${y}px) scale(${s})`;
      root.classList.toggle('is-zoomed', s > 1.001);
      level.textContent = Math.round(s * 100) + '%';
      btnOut.disabled = s <= MIN + 0.001;
      btnIn.disabled = s >= MAX - 0.001;
      btnReset.disabled = s <= MIN + 0.001;
      // Swap in the high-resolution image the first time the reader zooms.
      if (!hdLoaded && s > 1.2 && root.dataset.hd) {
        hdLoaded = true;
        const hd = new Image();
        hd.onload = () => { img.src = hd.src; };
        hd.src = root.dataset.hd;
      }
    }

    // Zoom by `factor`, keeping the point (cx, cy) in viewport coordinates fixed.
    function zoomAt(factor, cx, cy, animate) {
      const ns = Math.min(MAX, Math.max(MIN, s * factor));
      x = cx - (cx - x) * (ns / s);
      y = cy - (cy - y) * (ns / s);
      s = ns;
      apply(animate);
    }
    const center = () => { const { w, h } = size(); return [w / 2, h / 2]; };
    const local = (e) => { const r = viewport.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };

    btnIn.addEventListener('click', () => zoomAt(STEP, ...center(), true));
    btnOut.addEventListener('click', () => zoomAt(1 / STEP, ...center(), true));
    btnReset.addEventListener('click', () => { s = 1; x = 0; y = 0; apply(true); });

    viewport.addEventListener('wheel', (e) => {
      if (e.ctrlKey || e.metaKey) {           // Ctrl/Cmd + wheel, or trackpad pinch
        e.preventDefault();
        zoomAt(Math.exp(-e.deltaY * 0.01), ...local(e), false);
        return;
      }
      if (s <= 1.001) return;                  // not zoomed: let the page scroll
      const px = x, py = y;
      x -= e.deltaX; y -= e.deltaY;
      apply(false);
      if (x !== px || y !== py) e.preventDefault();  // at an edge: let the page scroll
    }, { passive: false });

    viewport.addEventListener('dblclick', (e) => {
      if (s > 1.001) { s = 1; x = 0; y = 0; apply(true); }
      else zoomAt(2.5, ...local(e), true);
    });

    // Drag to pan (mouse, pen, one finger when zoomed) and two-finger pinch.
    const pointers = new Map();
    let last = null, pinch = null;
    viewport.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      pointers.set(e.pointerId, local(e));
      viewport.setPointerCapture(e.pointerId);
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), s };
        last = null;
      } else if (s > 1.001) {
        last = local(e);
        root.classList.add('is-dragging');
      }
    });
    viewport.addEventListener('pointermove', (e) => {
      if (!pointers.has(e.pointerId)) return;
      pointers.set(e.pointerId, local(e));
      if (pinch && pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        const target = Math.min(MAX, Math.max(MIN, pinch.s * d / pinch.d));
        zoomAt(target / s, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, false);
      } else if (last) {
        const p = local(e);
        x += p[0] - last[0]; y += p[1] - last[1];
        last = p;
        apply(false);
      }
    });
    const end = (e) => {
      pointers.delete(e.pointerId);
      if (pointers.size < 2) pinch = null;
      if (pointers.size === 0) { last = null; root.classList.remove('is-dragging'); }
    };
    viewport.addEventListener('pointerup', end);
    viewport.addEventListener('pointercancel', end);

    root.addEventListener('keydown', (e) => {
      if (e.target !== root) return;
      const pan = 60;
      const keys = {
        '+': () => zoomAt(STEP, ...center(), true), '=': () => zoomAt(STEP, ...center(), true),
        '-': () => zoomAt(1 / STEP, ...center(), true), '0': () => { s = 1; x = 0; y = 0; apply(true); },
        ArrowLeft: () => { x += pan; apply(true); }, ArrowRight: () => { x -= pan; apply(true); },
        ArrowUp: () => { y += pan; apply(true); }, ArrowDown: () => { y -= pan; apply(true); },
      };
      if (keys[e.key] && (s > 1.001 || !e.key.startsWith('Arrow'))) { e.preventDefault(); keys[e.key](); }
    });

    // Keep the same relative view when the card is resized.
    let prevW = 0;
    new ResizeObserver(() => {
      const { w } = size();
      if (prevW && w !== prevW) { x *= w / prevW; y *= w / prevW; }
      prevW = w;
      apply(false);
    }).observe(viewport);

    apply(false);
  });
})();
