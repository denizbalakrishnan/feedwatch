// feed.js: the swipeable video card.
// Presentation only. It measures how long you stayed on a card and reports
// that one number, plus whether you tapped the heart. It knows nothing about
// the model.

const SWIPE_DISTANCE = 60; // pixels of upward drag that count as a swipe
const LIKE_PAUSE_MS = 700; // lets you see your dot move before the next card arrives

export class Feed {
  /**
   * onReaction(video, watchRatio, liked) is called once per card, the moment
   * you react. onNeedNext() is called when the card should be replaced.
   */
  constructor(root, { onReaction, onNeedNext, reducedMotion }) {
    this.card = root.querySelector('#card');
    this.face = root.querySelector('#cardFace');
    this.icon = root.querySelector('#cardIcon');
    this.title = root.querySelector('#cardTitle');
    this.heart = root.querySelector('#cardHeart');
    this.watchFill = root.querySelector('#watchFill');
    this.watchText = root.querySelector('#watchText');
    this.swipeHint = root.querySelector('#swipeHint');
    this.buttons = [root.querySelector('#nextButton'), root.querySelector('#likeButton')];
    this.onReaction = onReaction;
    this.onNeedNext = onNeedNext;
    this.reducedMotion = reducedMotion;

    this.video = null;
    this.elapsedMs = 0;
    this.locked = true; // true while no reaction is accepted
    this.onScreen = true;
    this.dragStartY = null;
    this.lastTick = performance.now();

    this.buttons[0].addEventListener('click', () => this.next());
    this.buttons[1].addEventListener('click', () => this.like());
    this.card.addEventListener('dblclick', () => this.like());
    this.listenForSwipes();
    document.addEventListener('keydown', (event) => this.onKey(event));
    new IntersectionObserver(([entry]) => { this.onScreen = entry.isIntersecting; }, { threshold: 0.5 })
      .observe(this.card);
    requestAnimationFrame((now) => this.tick(now));
  }

  /** Puts a new video on the card and starts its watch timer from zero. */
  show(video, colour, icon) {
    this.video = video;
    this.elapsedMs = 0;
    this.locked = false;
    this.face.style.setProperty('--card-colour', colour);
    this.face.style.setProperty('--card-angle', `${(video.index * 37) % 180}deg`);
    this.icon.textContent = icon;
    this.title.textContent = video.title;
    this.heart.classList.remove('on');
    for (const button of this.buttons) button.disabled = false;
    this.face.classList.remove('loading', 'entering');
    void this.face.offsetWidth; // forces the browser to restart the entrance animation
    this.face.classList.add('entering');
    this.drawWatch();
  }

  /** Shown when every video has been seen. */
  showEnd() {
    this.video = null;
    this.locked = true;
    for (const button of this.buttons) button.disabled = true;
    this.icon.textContent = '🏁';
    this.title.textContent = 'You have seen every video.';
    this.watchText.textContent = '';
  }

  /** Share of the video watched so far, from 0 to 1. The bar stops when the video ends. */
  watchRatio() {
    return Math.min(1, this.elapsedMs / (this.video.seconds * 1000));
  }

  next() {
    if (this.locked) return;
    this.locked = true;
    this.swipeHint.hidden = true; // you have found the way; no need to keep pointing
    this.onReaction(this.video, this.watchRatio(), false);
    this.onNeedNext();
  }

  like() {
    if (this.locked) return;
    this.locked = true;
    this.swipeHint.hidden = true;
    this.heart.classList.add('on');
    this.onReaction(this.video, this.watchRatio(), true);
    setTimeout(() => this.onNeedNext(), this.reducedMotion ? 0 : LIKE_PAUSE_MS);
  }

  /** The watch timer. It only runs while the card is on screen and the tab is visible. */
  tick(now) {
    const sinceLast = Math.min(now - this.lastTick, 100);
    this.lastTick = now;
    if (!this.locked && this.onScreen && document.visibilityState === 'visible') {
      this.elapsedMs += sinceLast;
      this.drawWatch();
    }
    requestAnimationFrame((next) => this.tick(next));
  }

  drawWatch() {
    const percent = Math.round(this.watchRatio() * 100);
    this.watchFill.style.width = `${percent}%`;
    this.watchText.textContent = percent >= 100 ? 'watched to the end' : `watched ${percent}%`;
  }

  listenForSwipes() {
    this.card.addEventListener('pointerdown', (event) => {
      this.dragStartY = event.clientY;
      this.card.setPointerCapture(event.pointerId);
    });
    this.card.addEventListener('pointermove', (event) => {
      if (this.dragStartY === null) return;
      const dragged = Math.min(0, event.clientY - this.dragStartY);
      this.face.style.transform = `translateY(${dragged}px)`;
    });
    const release = (event) => {
      if (this.dragStartY === null) return;
      const dragged = event.clientY - this.dragStartY;
      this.dragStartY = null;
      this.face.style.transform = '';
      if (dragged < -SWIPE_DISTANCE) this.next();
    };
    this.card.addEventListener('pointerup', release);
    this.card.addEventListener('pointercancel', release);
  }

  /** Arrow keys move to the next video and L likes, unless you are typing in a control or have scrolled away. */
  onKey(event) {
    const tag = event.target.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || tag === 'BUTTON' || tag === 'SUMMARY') return;
    if (!this.onScreen || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      this.next();
    } else if (event.key === 'l' || event.key === 'L') {
      this.like();
    }
  }
}
