// map.js: draws the taste map on a canvas.
// Presentation only. It is handed finished 2-D points and does none of the
// model's arithmetic; that all lives in mf.js and linalg.js.

const DOT_RADIUS = 3.2;
const MOVE_MS = 800; // how long your dot takes to glide to its new place
const HOVER_REACH = 14; // pixels: how close the pointer must be to name a dot
const FULL_TURN = 2 * Math.PI;

export class TasteMap {
  /**
   * theme holds four colours read from the CSS tokens, as 6-digit hex:
   * dot (a video before the reveal), you, text and background.
   */
  constructor(canvas, { colours, theme, reducedMotion, showUser = true, showLabels = true }) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.theme = theme;
    this.colours = colours;
    this.showUser = showUser; // false = no "you" dot, trail or line to the playing video
    this.showLabels = showLabels; // false = no category names on the map, even when revealed
    this.categoryLabels = [];
    this.titles = [];
    this.reducedMotion = reducedMotion;

    this.points = [];
    this.categories = [];
    this.centroids = [];
    this.reach = 1; // map units from the centre to the edge of the view
    this.revealed = false;
    this.shown = new Map(); // video index -> true (read as yes) or false (read as no)
    this.target = -1; // the video playing now
    this.marker = null; // one extra labelled point, or null
    this.userLabel = 'you';
    this.hovered = -1;
    this.user = [0, 0];
    this.trail = []; // every place your dot has settled, oldest first
    this.animating = false;

    canvas.addEventListener('pointermove', (event) => this.hoverAt(event));
    canvas.addEventListener('pointerdown', (event) => this.hoverAt(event));
    // A finger "leaves" the moment it lifts, which would wipe the name it just asked for, so only a mouse clears it.
    canvas.addEventListener('pointerleave', (event) => { if (event.pointerType === 'mouse') this.setHovered(-1); });
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  /**
   * Gives the map its video dots. categories is the hidden category of each
   * video and categoryLabels names them (both used only when revealed);
   * titles are shown when a dot is pointed at.
   */
  setVideos(points, { categories, categoryLabels, titles }) {
    this.points = points;
    this.categories = categories;
    this.categoryLabels = categoryLabels;
    this.titles = titles;
    this.hovered = -1;
    let furthest = 0;
    for (const point of points) furthest = Math.max(furthest, Math.hypot(point[0], point[1]));
    this.reach = furthest * 1.15 || 1;

    const sums = categoryLabels.map(() => [0, 0, 0]);
    points.forEach((point, item) => {
      const sum = sums[categories[item]];
      sum[0] += point[0];
      sum[1] += point[1];
      sum[2] += 1;
    });
    this.centroids = sums.map((sum) => [sum[0] / sum[2], sum[1] / sum[2]]);
    this.resize();
  }

  setRevealed(revealed) {
    this.revealed = revealed;
    this.draw();
  }

  setTarget(item) {
    this.target = item;
    this.hovered = -1; // a name left over from a tap should not sit on the map for ever
    this.draw();
  }

  /** Changes the word written beside the bright dot ("you", or "fan" during a fast-forward run). */
  setUserLabel(label) {
    this.userLabel = label;
    this.draw();
  }

  markShown(item, readAsYes) {
    this.shown.set(item, readAsYes);
  }

  /** Shows (or, with null, removes) one extra labelled marker, used for a brand-new video. */
  setMarker(point, label = '') {
    this.marker = point ? { point: [point[0], point[1]], label } : null;
    this.draw();
  }

  /** Puts your dot somewhere without animation (used once, at the start). */
  placeUser(point) {
    this.user = [point[0], point[1]];
    this.trail = [this.user];
    this.draw();
  }

  /** Wipes your ticks and your trail and puts your dot back, for "start again". */
  forgetUser(point) {
    this.shown.clear();
    this.placeUser(point);
  }

  /** Glides your dot to a new place and remembers the path. */
  moveUser(point) {
    this.moveFrom = [this.user[0], this.user[1]];
    this.moveTo = [point[0], point[1]];
    this.trail.push(this.moveTo);
    if (this.trail.length > 40) this.trail.shift();
    if (this.reducedMotion) {
      this.user = this.moveTo;
      this.draw();
      return;
    }
    this.moveStart = performance.now();
    if (!this.animating) {
      this.animating = true;
      requestAnimationFrame((now) => this.step(now));
    }
  }

  step(now) {
    const progress = Math.min(1, (now - this.moveStart) / MOVE_MS);
    const eased = 1 - Math.pow(1 - progress, 3); // fast at first, gentle at the end
    this.user = [
      this.moveFrom[0] + (this.moveTo[0] - this.moveFrom[0]) * eased,
      this.moveFrom[1] + (this.moveTo[1] - this.moveFrom[1]) * eased,
    ];
    this.draw();
    if (progress < 1) requestAnimationFrame((next) => this.step(next));
    else this.animating = false;
  }

  resize() {
    const box = this.canvas.getBoundingClientRect();
    const pixelRatio = window.devicePixelRatio || 1;
    this.width = box.width;
    this.height = box.height;
    this.canvas.width = Math.round(box.width * pixelRatio);
    this.canvas.height = Math.round(box.height * pixelRatio);
    this.context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    this.draw();
  }

  /** Map units to canvas pixels. The centre of the canvas is the all-zeros list: a person nothing is known about. */
  toScreen(point) {
    const scale = Math.min(this.width, this.height) / 2 / this.reach;
    let x = point[0];
    let y = point[1];
    const distance = Math.hypot(x, y);
    if (distance > this.reach * 0.97) { // keep a far-out dot on the canvas
      x *= (this.reach * 0.97) / distance;
      y *= (this.reach * 0.97) / distance;
    }
    return [this.width / 2 + x * scale, this.height / 2 - y * scale];
  }

  hoverAt(event) {
    const box = this.canvas.getBoundingClientRect();
    const pointerX = event.clientX - box.left;
    const pointerY = event.clientY - box.top;
    let nearest = -1;
    let nearestDistance = event.pointerType === 'mouse' ? HOVER_REACH : 2 * HOVER_REACH; // fingers are less exact
    for (let item = 0; item < this.points.length; item++) {
      const [x, y] = this.toScreen(this.points[item]);
      const distance = Math.hypot(x - pointerX, y - pointerY);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearest = item;
      }
    }
    this.setHovered(nearest);
  }

  setHovered(item) {
    if (item === this.hovered) return;
    this.hovered = item;
    this.draw();
  }

  draw() {
    if (!this.width) return;
    const context = this.context;
    context.clearRect(0, 0, this.width, this.height);
    this.drawVideos(context);
    if (this.revealed && this.showLabels) this.drawCategoryLabels(context);
    if (this.showUser) {
      this.drawTrail(context);
      this.drawTargetLine(context);
      this.drawUser(context);
    }
    if (this.marker) this.drawMarker(context);
    if (this.hovered !== -1) this.drawTooltip(context, this.hovered);
  }

  /** A white diamond with a label: a different shape from every other dot, so it does not rely on colour. */
  drawMarker(context) {
    const [x, y] = this.toScreen(this.marker.point);
    const reach = 9;
    context.beginPath();
    context.moveTo(x, y - reach);
    context.lineTo(x + reach, y);
    context.lineTo(x, y + reach);
    context.lineTo(x - reach, y);
    context.closePath();
    context.fillStyle = this.theme.text;
    context.fill();
    context.lineWidth = 3;
    context.strokeStyle = this.theme.background;
    context.stroke();
    context.font = '700 13px system-ui, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'alphabetic';
    context.lineWidth = 4;
    context.strokeText(this.marker.label, x, y - 15);
    context.fillStyle = this.theme.text;
    context.fillText(this.marker.label, x, y - 15);
  }

  drawVideos(context) {
    for (let item = 0; item < this.points.length; item++) {
      const [x, y] = this.toScreen(this.points[item]);
      const answer = this.shown.get(item);
      context.beginPath();
      context.arc(x, y, DOT_RADIUS, 0, FULL_TURN);
      context.fillStyle = this.revealed ? this.colours[this.categories[item]] : this.theme.dot;
      context.globalAlpha = answer === false ? 0.3 : this.revealed ? 0.95 : 0.7;
      context.fill();
      if (answer === true) { // videos you said yes to get a white ring
        context.globalAlpha = 1;
        context.lineWidth = 2;
        context.strokeStyle = this.theme.text;
        context.stroke();
      }
    }
    context.globalAlpha = 1;
  }

  drawCategoryLabels(context) {
    context.font = '600 13px system-ui, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.lineJoin = 'round';
    this.centroids.forEach((centroid, category) => {
      const [x, y] = this.toScreen(centroid);
      context.lineWidth = 4;
      context.strokeStyle = this.theme.background;
      context.strokeText(this.categoryLabels[category], x, y - 16);
      context.fillStyle = this.theme.text;
      context.fillText(this.categoryLabels[category], x, y - 16);
    });
  }

  drawTrail(context) {
    if (this.trail.length < 2) return;
    context.lineWidth = 2;
    context.strokeStyle = this.theme.you;
    for (let step = 1; step < this.trail.length; step++) {
      const isLast = step === this.trail.length - 1;
      const [fromX, fromY] = this.toScreen(this.trail[step - 1]);
      const [toX, toY] = this.toScreen(isLast ? this.user : this.trail[step]);
      context.globalAlpha = 0.15 + 0.55 * (step / (this.trail.length - 1));
      context.beginPath();
      context.moveTo(fromX, fromY);
      context.lineTo(toX, toY);
      context.stroke();
    }
    context.globalAlpha = 1;
  }

  drawTargetLine(context) {
    if (this.target === -1) return;
    const [userX, userY] = this.toScreen(this.user);
    const [targetX, targetY] = this.toScreen(this.points[this.target]);
    context.setLineDash([4, 5]);
    context.lineWidth = 1.5;
    context.strokeStyle = this.theme.text;
    context.globalAlpha = 0.55;
    context.beginPath();
    context.moveTo(userX, userY);
    context.lineTo(targetX, targetY);
    context.stroke();
    context.globalAlpha = 1;
    context.setLineDash([]);
    context.beginPath();
    context.arc(targetX, targetY, DOT_RADIUS + 5, 0, FULL_TURN);
    context.stroke();
  }

  drawUser(context) {
    const [x, y] = this.toScreen(this.user);
    const glow = context.createRadialGradient(x, y, 2, x, y, 30);
    glow.addColorStop(0, `${this.theme.you}99`); // the same colour at 60% strength
    glow.addColorStop(1, `${this.theme.you}00`); // fading to fully see-through
    context.fillStyle = glow;
    context.beginPath();
    context.arc(x, y, 30, 0, FULL_TURN);
    context.fill();
    context.beginPath();
    context.arc(x, y, 7, 0, FULL_TURN);
    context.fillStyle = this.theme.text;
    context.fill();
    context.lineWidth = 3;
    context.strokeStyle = this.theme.you;
    context.stroke();
    context.font = '700 13px system-ui, sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'alphabetic';
    context.lineWidth = 4;
    context.strokeStyle = this.theme.background;
    context.strokeText(this.userLabel, x, y - 14);
    context.fillStyle = this.theme.you;
    context.fillText(this.userLabel, x, y - 14);
  }

  drawTooltip(context, item) {
    const [x, y] = this.toScreen(this.points[item]);
    const text = this.titles[item];
    context.font = '500 12px system-ui, sans-serif';
    const boxWidth = context.measureText(text).width + 16;
    const boxX = Math.min(Math.max(4, x - boxWidth / 2), this.width - boxWidth - 4);
    const boxY = y > 40 ? y - 34 : y + 12;
    context.fillStyle = this.theme.background;
    context.strokeStyle = this.theme.dot;
    context.lineWidth = 1;
    context.beginPath();
    context.rect(boxX, boxY, boxWidth, 22);
    context.fill();
    context.stroke();
    context.fillStyle = this.theme.text;
    context.textAlign = 'left';
    context.textBaseline = 'middle';
    context.fillText(text, boxX + 8, boxY + 11);
  }
}
