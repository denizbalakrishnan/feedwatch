// ui.js: wires the page together.
// It asks mf.js and linalg.js for every number and hands the results to the
// feed card (feed.js) and the map (map.js). No model arithmetic happens here.

import {
  DEFAULT_SETTINGS, UNSEEN, LIKE_AS_WATCH_RATIO, trainALS, foldInUser, scoreItems, bestUnseen, preferenceOf,
  confidenceOf,
} from './mf.js';
import { principalAxes, projectOntoAxes } from './linalg.js';
import { pickEpsilonGreedy, pickThompson, categoryMix } from './explore.js';
import { makeRandom } from './random.js';
import { loadCrowd } from './data.js';
import { TasteMap } from './map.js';
import { Feed } from './feed.js';

// The Okabe-Ito palette: eight colours chosen to stay distinct for colour-blind viewers.
const CATEGORY_COLOURS = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#BBBBBB'];
const NUDGE_AFTER = 3; // reactions before we point at the reveal switch
const MAX_EPSILON = 0.5; // "random sometimes" at full strength: every other video is random
const MAX_SPREAD = 2; // Thompson sampling at full strength: twice the model's real uncertainty
const RECENT_COUNT = 20; // how many recommendations the bubble meter looks back over
const AUTOPILOT_SWIPES = 30;
const AUTOPILOT_STEP_MS = 90;

const settings = { ...DEFAULT_SETTINGS };
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const find = (id) => document.getElementById(id);

const state = {
  crowd: null,
  itemFactors: null,
  itemAppeal: null,
  axes: null,
  userScale: 1,
  myWatchRatios: null, // one cell per video: UNSEEN, or the watch ratio you gave it
  foldIn: null, // the latest fold-in result: your vector and how sure the model is of it
  myVector: null,
  shownOrder: [], // video indexes in the order you saw them
  currentVideo: null,
  recentCategories: [], // hidden categories of the last few recommendations, for the bubble meter
  exploreMode: 'greedy', // 'greedy', 'epsilon' or 'thompson'
  exploreAmount: 0.5, // the slider, from 0 to 1
  random: makeRandom(7),
};

let tasteMap;
let feed;

/** Trains the model in this tab, lays out the map and deals the first card. */
function start() {
  const crowd = loadCrowd();
  const began = performance.now();
  const model = trainALS(crowd.watchRatios, settings);
  const trainSeconds = (performance.now() - began) / 1000;

  state.crowd = crowd;
  state.itemFactors = model.itemFactors;
  state.itemAppeal = model.itemAppeal;
  state.axes = principalAxes(model.itemFactors, 2).axes;
  state.myWatchRatios = new Float32Array(crowd.videos.length).fill(UNSEEN);
  refoldUser(); // with no history this gives all zeros: no taste known yet

  const videoPoints = model.itemFactors.map((factor) => projectOntoAxes(factor, state.axes));
  const crowdPoints = model.userFactors.map((factor) => projectOntoAxes(factor, state.axes));
  state.userScale = typicalDistance(videoPoints) / typicalDistance(crowdPoints);

  tasteMap = new TasteMap(find('map'), {
    colours: CATEGORY_COLOURS,
    categoryLabels: crowd.categories.map((category) => `${category.icon} ${category.name}`),
    titles: crowd.videos.map((video) => video.title),
    theme: {
      dot: cssToken('--color-dot'),
      you: cssToken('--color-you'),
      text: cssToken('--color-text'),
      background: cssToken('--color-background'),
    },
    reducedMotion,
  });
  tasteMap.setVideos(videoPoints, crowd.videos.map((video) => video.category));
  tasteMap.placeUser(mapPointOf(state.myVector));

  feed = new Feed(find('feed'), { onReaction: recordReaction, onNeedNext: dealNextCard, reducedMotion });
  find('reveal').addEventListener('change', (event) => setRevealed(event.target.checked));
  wireExplorationPanel();

  const cellCount = (crowd.watchRatios.length * crowd.videos.length).toLocaleString('en');
  find('mapCaption').textContent =
    `Every dot is one video. The bright dot is you. The model was trained ${trainSeconds.toFixed(1)} s ago, ` +
    `in this tab, on ${cellCount} watch times from ${crowd.sourceLabel}, and on nothing else.`;
  renderVectorPanel();
  dealNextCard();
  // Only now is there a model to take apart, so only now do the controls wake up.
  find('reveal').disabled = false;
  find('lab').hidden = false;
}

/**
 * The model only fixes the product "person . video", not the size of each
 * side, so people's vectors come out shorter than videos'. We draw both at
 * the same overall size: this is the typical distance from the map's centre.
 */
function typicalDistance(points) {
  let total = 0;
  for (const point of points) total += point[0] * point[0] + point[1] * point[1];
  return Math.sqrt(total / points.length);
}

/** Reads one colour from the tokens at the top of style.css, so the canvas and the page cannot drift apart. */
function cssToken(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function mapPointOf(userVector) {
  const flat = projectOntoAxes(userVector, state.axes);
  return [flat[0] * state.userScale, flat[1] * state.userScale];
}

/** Called once per card. This is the whole of what the model learns about you. */
function recordReaction(video, watchRatio, liked) {
  const ratio = liked ? LIKE_AS_WATCH_RATIO : watchRatio;
  state.myWatchRatios[video.index] = ratio;
  state.shownOrder.push(video.index);
  refoldUser();

  const readAsYes = preferenceOf(ratio, settings) === 1;
  tasteMap.markShown(video.index, readAsYes);
  tasteMap.moveUser(mapPointOf(state.myVector));
  find('map').setAttribute('aria-label',
    `Taste map of ${state.crowd.videos.length} videos. Your dot has moved ${state.shownOrder.length} times.`);

  const told = liked ? 'liked' : `watched ${Math.round(watchRatio * 100)}%`;
  const verdict = liked ? 'a firm yes' : readAsYes ? 'a yes' : watchRatio < 0.2 ? 'a firm no' : 'a no';
  find('reading').textContent =
    `All the model was told: video ${video.index + 1}, ${told}. It reads that as ${verdict} and moves you.`;

  renderVectorPanel();
  if (state.shownOrder.length === NUDGE_AFTER && !find('reveal').checked) find('revealLabel').classList.add('nudge');
}

/** Solves your vector again from everything you have reacted to so far. */
function refoldUser() {
  state.foldIn = foldInUser(state.itemFactors, state.itemAppeal, state.myWatchRatios, settings);
  state.myVector = state.foldIn.vector;
}

/** Chooses the next video with whichever rule the exploration panel is set to. */
function chooseNextVideo() {
  if (state.exploreMode === 'thompson') {
    const spread = MAX_SPREAD * state.exploreAmount;
    return pickThompson(state.foldIn, state.itemFactors, state.itemAppeal, state.myWatchRatios, spread, state.random);
  }
  const scores = scoreItems(state.myVector, state.itemFactors, state.itemAppeal);
  if (state.exploreMode === 'epsilon') {
    return pickEpsilonGreedy(scores, state.myWatchRatios, MAX_EPSILON * state.exploreAmount, state.random);
  }
  return bestUnseen(scores, state.myWatchRatios);
}

/** Puts the chosen video on the card and points at it on the map. */
function dealNextCard() {
  const item = chooseNextVideo();
  if (item === -1) {
    state.currentVideo = null;
    feed.showEnd();
    tasteMap.setTarget(-1);
    return;
  }
  const video = state.crowd.videos[item];
  state.currentVideo = video;
  state.recentCategories.push(video.category);
  if (state.recentCategories.length > RECENT_COUNT) state.recentCategories.shift();
  feed.show(video, CATEGORY_COLOURS[video.category], state.crowd.categories[video.category].icon);
  tasteMap.setTarget(item);
  renderBubbleMeter();
}

/** Shows how varied your recent recommendations were: 0% is a sealed bubble. */
function renderBubbleMeter() {
  const mix = categoryMix(state.recentCategories, state.crowd.categories.length);
  const percent = Math.round(mix * 100);
  find('bubbleFill').style.width = `${percent}%`;
  const verdict = mix < 0.3 ? 'A bubble: nearly everything is one topic.'
    : mix < 0.65 ? 'Narrowing: a few topics dominate.'
      : 'Varied: many topics are getting through.';
  find('bubbleText').textContent =
    `Variety of your last ${state.recentCategories.length} recommendations: ${percent}%. ${verdict}`;
}

/** Connects the exploration panel's controls to the feed. */
function wireExplorationPanel() {
  for (const radio of document.querySelectorAll('input[name="explore"]')) {
    radio.addEventListener('change', () => {
      state.exploreMode = radio.value;
      renderExploreAmount();
    });
  }
  find('exploreAmount').addEventListener('input', (event) => {
    state.exploreAmount = Number(event.target.value) / 100;
    renderExploreAmount();
  });
  const topicPicker = find('autopilotTopic');
  state.crowd.categories.forEach((category, index) => {
    const option = make('option', '', `${category.icon} ${category.name}`);
    option.value = String(index);
    topicPicker.append(option);
  });
  find('autopilotButton').addEventListener('click', () => runAutopilot(Number(topicPicker.value), AUTOPILOT_SWIPES));
  find('resetButton').addEventListener('click', startAgain);
  renderExploreAmount();
}

function renderExploreAmount() {
  find('exploreAmountRow').hidden = state.exploreMode === 'greedy';
  find('exploreAmountText').textContent = state.exploreMode === 'epsilon'
    ? `${Math.round(MAX_EPSILON * state.exploreAmount * 100)}% of videos are picked at random`
    : `sampling at ${(MAX_SPREAD * state.exploreAmount).toFixed(1)} times the model's own uncertainty`;
}

/**
 * Fast-forwards the feed with a simple pretend viewer: they like every video
 * of one topic and skip everything else after a tenth of it. The model is
 * still only told watch ratios; the topic is the pretend viewer's secret.
 */
function runAutopilot(likedCategory, swipesLeft) {
  if (swipesLeft === 0 || !state.currentVideo) return;
  const video = state.currentVideo;
  const liked = video.category === likedCategory;
  recordReaction(video, 0.1, liked);
  dealNextCard();
  setTimeout(() => runAutopilot(likedCategory, swipesLeft - 1), reducedMotion ? 0 : AUTOPILOT_STEP_MS);
}

/** Forgets everything you did, as if a brand-new person had opened the feed. */
function startAgain() {
  state.myWatchRatios.fill(UNSEEN);
  state.shownOrder = [];
  state.recentCategories = [];
  refoldUser();
  tasteMap.forgetUser(mapPointOf(state.myVector));
  find('reading').textContent = 'Started again as a brand-new person.';
  renderVectorPanel();
  dealNextCard();
}

function setRevealed(revealed) {
  tasteMap.setRevealed(revealed);
  find('revealLabel').classList.remove('nudge');
  find('revealNote').hidden = !revealed;
}

/** Fills the "Your vector" panel with the real numbers behind the map. */
function renderVectorPanel() {
  const bars = find('vectorBars');
  bars.replaceChildren();
  let largest = 0.5;
  for (const value of state.myVector) largest = Math.max(largest, Math.abs(value));
  state.myVector.forEach((value, index) => {
    const row = make('div', 'vector-row');
    row.append(make('span', 'vector-name', `${index + 1}`));
    const track = make('span', 'vector-track');
    const fill = make('span', value < 0 ? 'vector-fill negative' : 'vector-fill');
    fill.style.width = `${(Math.abs(value) / largest) * 50}%`;
    track.append(fill);
    row.append(track, make('span', 'vector-value', value.toFixed(2)));
    bars.append(row);
  });

  const evidence = find('evidenceRows');
  evidence.replaceChildren();
  for (const item of state.shownOrder.slice(-8).reverse()) {
    const ratio = state.myWatchRatios[item];
    const row = make('tr');
    row.append(
      make('td', '', `video ${item + 1}`),
      make('td', '', ratio === LIKE_AS_WATCH_RATIO ? 'liked (counts as 2.00)' : ratio.toFixed(2)),
      make('td', '', preferenceOf(ratio, settings) === 1 ? 'yes (1)' : 'no (0)'),
      make('td', '', confidenceOf(ratio, settings).toFixed(1)),
    );
    evidence.append(row);
  }
  find('evidenceEmpty').hidden = state.shownOrder.length > 0;

  const scores = scoreItems(state.myVector, state.itemFactors, state.itemAppeal);
  const unseen = [];
  for (let item = 0; item < scores.length; item++) if (state.myWatchRatios[item] === UNSEEN) unseen.push(item);
  unseen.sort((a, b) => scores[b] - scores[a]);
  const upNext = find('upNext');
  upNext.replaceChildren();
  for (const item of unseen.slice(0, 5)) upNext.append(make('li', '', `video ${item + 1}: score ${scores[item].toFixed(2)}`));
}

function make(tag, className = '', text = '') {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

// Let the browser paint the empty page first, then train.
requestAnimationFrame(() => setTimeout(start, 0));
