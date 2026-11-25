// ui.js: wires the whole page together.
// It asks mf.js, linalg.js and explore.js for every number and hands the
// results to the feed card (feed.js), the map (map.js) and the chart
// (lab.js). Training runs in train.worker.js. All the training and solving
// is done by the maths files. The only sums written out here are small ones
// for display (a dot product, an average distance).

import {
  DEFAULT_SETTINGS, UNSEEN, PASSED, LIKE_AS_WATCH_RATIO, foldInUser, scoreItems, bestUnseen, preferenceOf, confidenceOf,
} from './mf.js';
import { dot, norm, principalAxes, projectOntoAxes, procrustes2D, applyMatrix2D } from './linalg.js';
import { pickEpsilonGreedy, pickThompson, categoryMix } from './explore.js';
import { TWO_TOWER_DEFAULTS, tagsToVector, embedNewItem, embedUser } from './twotower.js';
import { neighbourAgreement } from './sim.js';
import { makeRandom } from './random.js';
import { DEFAULT_CROWD, describeVideos, loadRealCrowd } from './data.js';
import { TasteMap } from './map.js';
import { Feed } from './feed.js';
import { drawErrorChart } from './lab.js';

// Seven colours from the Okabe-Ito palette, chosen to stay distinct for colour-blind viewers, plus a
// light grey (the palette's own eighth colour is black, which would vanish on our dark page).
const CATEGORY_COLOURS = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#BBBBBB'];
const REAL_COLOURS = [...CATEGORY_COLOURS, '#882255', '#44AA99']; // the real data has ten categories; two more from Paul Tol's "muted" set
// The real grid holds 1s and 0s, so every cell counts equally (confidenceGain 0), and 16 numbers per video.
const REAL_SETTINGS = { ...DEFAULT_SETTINGS, factorCount: 16, confidenceGain: 0, sweepCount: 8 };
const NO_TURN = [[1, 0], [0, 1]];
const NUDGE_AFTER = 3; // reactions before we point at the reveal switch
const MAX_EPSILON = 0.5; // "random sometimes" at full strength: every other video is random
const MAX_SPREAD = 2; // Thompson sampling at full strength: twice the spread the fold-in itself gives
const RECENT_COUNT = 20; // how many recommendations the bubble meter looks back over
const AUTOPILOT_SWIPES = 30;
const AUTOPILOT_STEP_MS = 90;

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const find = (id) => document.getElementById(id);

/** Reads one colour from the tokens at the top of style.css, so the canvas and the page cannot drift apart. */
function cssToken(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const theme = {
  dot: cssToken('--color-dot'),
  you: cssToken('--color-you'),
  text: cssToken('--color-text'),
  background: cssToken('--color-background'),
};

const state = {
  // the trained model and how it is drawn
  settings: { ...DEFAULT_SETTINGS }, // the settings the current model was trained with
  trainedCrowd: '', // which crowd it was trained on, as text, to spot a change
  jobId: 0, // goes up with every training request; late answers to old requests are ignored
  errorHistory: [],
  videos: [],
  categories: [],
  itemFactors: null,
  itemAppeal: null,
  axes: null, // the two directions of the model's space that the map shows
  turn: NO_TURN, // the Procrustes rotation that keeps a retrained map facing the same way
  videoPoints: null,
  userScale: 1,

  // the second model (two-tower); null while it is still training
  tower: null,
  towerItemVectors: null,
  towerView: null, // { axes, turn } for the second model's own map
  towerUserVector: null,

  // you
  reactions: new Map(), // video index -> { watchRatio, liked }, in the order you reacted
  shownMarks: null, // one cell per video: UNSEEN, or 1 once it has been on your card
  evidence: null, // one cell per video: what the model is told (depends on the signal choice)
  foldIn: null, // the latest fold-in result: your vector and how sure the model is of it
  myVector: null,
  currentVideo: null,
  recentCategories: [], // hidden categories of the last few recommendations, for the bubble meter
  isPretendFan: false, // true after a fast-forward run: the dot is then the pretend viewer, not you

  // lab choices
  signal: 'both', // 'both', 'likes' or 'watch'
  exploreMode: 'greedy', // 'greedy', 'epsilon' or 'thompson'
  exploreAmount: 0.25, // the slider, from 0 to 1 (0.25 = strength 0.5: explores, but still mostly serves your taste)
  random: makeRandom(7),
};

const tasteMap = new TasteMap(find('map'), { colours: CATEGORY_COLOURS, theme, reducedMotion });
const towerMap = new TasteMap(find('towerMap'), { colours: CATEGORY_COLOURS, theme, reducedMotion, showUser: false });
towerMap.setRevealed(true); // the second model is given the tags, so its map has nothing to hide
const realMap = new TasteMap(find('realMap'), { colours: REAL_COLOURS, theme, reducedMotion, showUser: false, showLabels: false });
realMap.setRevealed(true);
let realCrowd = null; // the real-data grid, fetched the first time it is asked for
const feed = new Feed(find('feed'), { onReaction: recordReaction, onNeedNext: dealNextCard, reducedMotion });
const trainer = new Worker(new URL('./train.worker.js', import.meta.url), { type: 'module' });
trainer.onmessage = (event) => onTrainerMessage(event.data);

// ---------- training ----------

/** Reads the lab's controls and asks the worker for a model. */
function requestTraining() {
  state.jobId += 1;
  state.errorHistory = [];
  find('trainButton').disabled = true;
  find('trainStatus').textContent = 'Training…';
  trainer.postMessage({
    jobId: state.jobId,
    crowdSettings: {
      ...DEFAULT_CROWD,
      userCount: Number(find('crowdUsers').value),
      topicCount: Number(find('crowdTopics').value),
      noise: Number(find('crowdNoise').value) / 100,
      sparsity: Number(find('crowdSparsity').value) / 100,
    },
    modelSettings: {
      ...DEFAULT_SETTINGS,
      factorCount: Number(document.querySelector('input[name="factorCount"]:checked').value),
    },
  });
}

function onTrainerMessage(message) {
  if (message.type === 'realSweep') {
    find('realStatus').textContent = `Training: sweep ${message.sweep} of ${REAL_SETTINGS.sweepCount}`;
    return;
  }
  if (message.type === 'real') {
    showRealModel(message);
    return;
  }
  if (message.jobId !== state.jobId) return; // an answer to a request we have since replaced
  if (message.type === 'sweep') {
    state.errorHistory.push(message);
    find('trainStatus').textContent = `Training: sweep ${message.sweep} of ${DEFAULT_SETTINGS.sweepCount}`;
    drawErrorChart(find('errorChart'), state.errorHistory, DEFAULT_SETTINGS.sweepCount, theme);
    return;
  }
  if (message.type === 'tower') showTower(message);
  else showTrainedModel(message);
}

/** Takes a freshly trained model from the worker and puts it on the page. */
function showTrainedModel(result) {
  const crowdAsText = JSON.stringify(result.crowdSettings);
  const sameCrowd = crowdAsText === state.trainedCrowd;
  state.trainedCrowd = crowdAsText;
  state.settings = result.modelSettings;
  state.itemFactors = result.itemFactors;
  state.itemAppeal = result.itemAppeal;
  state.tower = null; // the second model for this crowd arrives in a moment
  find('newVideoButton').disabled = true;
  find('towerCaption').textContent = 'Training the second model…';

  // Flatten the video vectors to 2-D. If these are the same videos as before,
  // turn the new layout to face the same way as the old one.
  state.axes = principalAxes(result.itemFactors, 2).axes;
  let points = result.itemFactors.map((factor) => projectOntoAxes(factor, state.axes));
  state.turn = sameCrowd && state.videoPoints ? procrustes2D(points, state.videoPoints) : NO_TURN;
  points = points.map((point) => applyMatrix2D(state.turn, point));
  state.videoPoints = points;
  const crowdPoints = result.userFactors.map((factor) => projectOntoAxes(factor, state.axes));
  state.userScale = typicalDistance(points) / typicalDistance(crowdPoints);

  if (!sameCrowd) {
    const described = describeVideos(result.itemTopics, result.crowdSettings.topicCount);
    state.videos = described.videos;
    state.categories = described.categories;
    fillTopicPicker();
  }
  tasteMap.setVideos(points, {
    categories: state.videos.map((video) => video.category),
    categoryLabels: state.categories.map((category) => `${category.icon} ${category.name}`),
    titles: state.videos.map((video) => video.title),
  });

  if (sameCrowd) { // same videos, new model: keep your history and work out your vector again
    rebuildEvidence();
    tasteMap.placeUser(mapPointOf(state.myVector));
    renderVectorPanel();
  } else {
    forgetPerson();
    dealNextCard();
  }
  describeTraining(result);
  find('trainButton').disabled = false;
  find('reveal').disabled = false;
  find('lab').hidden = false;
  drawErrorChart(find('errorChart'), state.errorHistory, DEFAULT_SETTINGS.sweepCount, theme);
}

/** Writes the captions that report what was just trained and how well it did. */
function describeTraining(result) {
  const crowd = result.crowdSettings;
  const size = result.modelSettings.factorCount;
  for (const node of document.querySelectorAll('.factor-count')) node.textContent = String(size);

  find('mapCaption').textContent =
    `The model was trained in ${result.seconds.toFixed(1)} s, in this tab, on ` +
    `${result.trainedCells.toLocaleString('en')} watch times from a simulated crowd of ` +
    `${crowd.userCount.toLocaleString('en')} people and ${crowd.itemCount} videos, and on nothing else.`;
  find('trainStatus').textContent = `Trained in ${result.seconds.toFixed(1)} s.`;

  const last = state.errorHistory[state.errorHistory.length - 1];
  find('errorText').textContent =
    `One watch time in ten was hidden from training. After the last sweep the model's average squared miss was ` +
    `${last.trainingError.toFixed(3)} on watch times it trained on and ${last.heldOutError.toFixed(3)} on the ` +
    `${result.hiddenCells.toLocaleString('en')} hidden ones. A large gap means it has memorised noise.`;

  const topics = state.videos.map((video) => video.category);
  const inModel = Math.round(100 * neighbourAgreement(state.itemFactors, topics));
  const onMap = Math.round(100 * neighbourAgreement(state.videoPoints, topics));
  find('recoveryText').textContent =
    `A video's nearest neighbour shares its hidden topic ${inModel}% of the time in the model's ${size} dimensions ` +
    `and ${onMap}% of the time on the flat map. Guessing would score about ${Math.round(100 / crowd.topicCount)}%.`;
}

/**
 * People's vectors and videos' vectors come out at different typical
 * lengths. For drawing only, we scale your dot so that the crowd and the
 * videos would sit at the same overall distance from the map's centre.
 * This returns that typical distance for a set of points.
 */
function typicalDistance(points) {
  let total = 0;
  for (const point of points) total += point[0] * point[0] + point[1] * point[1];
  return Math.sqrt(total / points.length) || 1;
}

function mapPointOf(userVector) {
  const flat = applyMatrix2D(state.turn, projectOntoAxes(userVector, state.axes));
  return [flat[0] * state.userScale, flat[1] * state.userScale];
}

// ---------- the second model (two-tower) ----------

/** Takes the trained two-tower model from the worker and lays out its own map. */
function showTower(result) {
  state.tower = result.tower;
  state.towerItemVectors = result.itemVectors;
  const axes = principalAxes(result.itemVectors, 2).axes;
  let points = result.itemVectors.map((vector) => projectOntoAxes(vector, axes));
  const turn = procrustes2D(points, state.videoPoints); // same videos as the main map, so face the same way
  points = points.map((point) => applyMatrix2D(turn, point));
  state.towerView = { axes, turn };

  towerMap.setVideos(points, {
    categories: state.videos.map((video) => video.category),
    categoryLabels: state.categories.map((category) => `${category.icon} ${category.name}`),
    titles: state.videos.map((video) => video.title),
  });
  towerMap.setMarker(null);
  fillTagChoices();
  find('newVideoFirst').textContent = '';
  find('newVideoSecond').textContent = '';
  find('newVideoButton').disabled = false;

  find('towerCaption').textContent =
    `The same ${state.videos.length} videos, placed by the two-tower model and turned to face the same way as the ` +
    `map at the top. It was trained in ${result.seconds.toFixed(1)} s (${result.stepCount} steps of ` +
    `${TWO_TOWER_DEFAULTS.batchSize} examples) and was given each video's tag, so here the colours are no secret.`;
  const percent = (share) => `${Math.round(100 * share)}%`;
  find('precisionText').textContent = result.firstPrecision === null
    ? `One watch time in ten was hidden from both models during training, but with this crowd nobody has ` +
      `${result.topCount} hidden videos to rank, so neither model can be marked. Lower the gaps and train again.`
    : `One watch time in ten was hidden from both models during training. Each model then ranked every person's ` +
      `hidden videos. Of each person's top ${result.topCount}, the share they really watched: first model ` +
      `${percent(result.firstPrecision)}, two-tower model ${percent(result.towerPrecision)} ` +
      `(${result.peopleMarked.toLocaleString('en')} people marked). Picking at random would score about ` +
      `${percent(result.yesShare)}. The comparison favours the two-tower model: each video's tag here is its topic, ` +
      'which the first model has to work out for itself. We include the two-tower model because it can place a brand-new video.';
  renderTowerPanel();
}

function towerPointOf(vector) {
  return applyMatrix2D(state.towerView.turn, projectOntoAxes(vector, state.towerView.axes));
}

/** The videos you have said yes to. This list is all the two-tower model knows about you. */
function likedHistory() {
  const liked = [];
  for (const item of state.reactions.keys()) {
    if (preferenceOf(state.evidence[item], state.settings) === 1) liked.push(item);
  }
  return liked;
}

/** The best-scoring videos you have not been shown, best first. */
function topUnseen(scores, count) {
  const unseen = [];
  for (let item = 0; item < scores.length; item++) if (state.shownMarks[item] === UNSEEN) unseen.push(item);
  unseen.sort((a, b) => scores[b] - scores[a]);
  return unseen.slice(0, count);
}

/** Works out where the second model puts you, and what each model would play you next. */
function renderTowerPanel() {
  if (!state.tower) return;
  state.towerUserVector = embedUser(state.tower, likedHistory());

  const firstScores = scoreItems(state.myVector, state.itemFactors, state.itemAppeal);
  const towerScores = state.towerItemVectors.map((vector) => dot(state.towerUserVector, vector));
  for (const [listId, scores] of [['firstTop', firstScores], ['towerTop', towerScores]]) {
    const list = find(listId);
    list.replaceChildren();
    for (const item of topUnseen(scores, 5)) {
      const video = state.videos[item];
      list.append(make('li', '', `${state.categories[video.category].icon} ${video.title}`));
    }
  }
}

/** The cold-start demo: a video nobody has watched, described only by the tags you ticked. */
function uploadNewVideo() {
  const tags = [...document.querySelectorAll('input[name="newVideoTag"]:checked')].map((box) => Number(box.value));
  if (tags.length === 0) {
    find('newVideoFirst').textContent = 'Tick at least one tag first.';
    find('newVideoSecond').textContent = '';
    return;
  }
  const vector = embedNewItem(state.tower, tagsToVector(tags, state.categories.length));
  towerMap.setMarker(towerPointOf(vector), 'new video');

  const nearest = state.towerItemVectors
    .map((known, item) => [dot(vector, known) / (norm(vector) * norm(known)), item])
    .sort((a, b) => b[0] - a[0])
    .slice(0, 3)
    .map(([, item]) => state.videos[item].title);
  const scoreForYou = dot(state.towerUserVector, vector);
  let ahead = 0;
  let unseen = 0;
  state.towerItemVectors.forEach((known, item) => {
    if (state.shownMarks[item] !== UNSEEN) return;
    unseen += 1;
    if (dot(state.towerUserVector, known) > scoreForYou) ahead += 1;
  });

  find('newVideoFirst').textContent =
    'First model: cannot place it. Nobody has watched this video, so its column in the watch grid is empty. ' +
    'It has no numbers, so it cannot be scored and would never be recommended.';
  find('newVideoSecond').textContent =
    `Two-tower model: placed at once from its tags alone (the white diamond). Its nearest neighbours are ` +
    `${nearest.join(', ')}. For you it scores ${scoreForYou.toFixed(2)}, which would put it at number ` +
    `${ahead + 1} of ${unseen + 1} in your feed.`;
}

/** One tick-box per tag, for describing a brand-new video. */
function fillTagChoices() {
  const holder = find('newVideoTags');
  for (const old of holder.querySelectorAll('label')) old.remove();
  state.categories.forEach((category, index) => {
    const label = make('label');
    const box = make('input');
    box.type = 'checkbox';
    box.name = 'newVideoTag';
    box.value = String(index);
    label.append(box, make('span', '', `${category.icon} ${category.name}`));
    holder.append(label);
  });
}

// ---------- real people ----------

/** Fetches the real grid (once) and asks the worker to train the first model on it. */
async function trainOnRealPeople() {
  find('realButton').disabled = true;
  find('realStatus').textContent = 'Fetching the grid…';
  try {
    realCrowd = realCrowd || await loadRealCrowd();
  } catch (error) {
    find('realStatus').textContent = 'The data file could not be loaded. Check your connection and try again.';
    find('realButton').disabled = false;
    return;
  }
  find('realStatus').textContent = 'Training…';
  trainer.postMessage({ realGrid: realCrowd.grid, settings: REAL_SETTINGS });
}

/** Marks the model trained on real people against the real categories, and draws its map. */
function showRealModel(result) {
  const axes = principalAxes(result.itemFactors, 2).axes;
  const points = result.itemFactors.map((factor) => projectOntoAxes(factor, axes));
  realMap.setVideos(points, {
    categories: realCrowd.videoCategory,
    categoryLabels: realCrowd.categories,
    titles: realCrowd.videoCategory.map((category, video) => `${realCrowd.categories[category]} video ${video + 1}`),
  });
  const legend = find('realLegend');
  legend.replaceChildren();
  realCrowd.categories.forEach((name, category) => {
    const entry = make('li', '', name);
    const swatch = make('span', 'swatch');
    swatch.style.background = REAL_COLOURS[category];
    entry.prepend(swatch);
    legend.append(entry);
  });

  const inModel = neighbourAgreement(result.itemFactors, realCrowd.videoCategory);
  const onMap = neighbourAgreement(points, realCrowd.videoCategory);
  const chance = 1 / realCrowd.categories.length;
  find('realStatus').textContent = `Trained in ${result.seconds.toFixed(1)} s.`;
  find('realText').textContent =
    `Marked against the real categories: a video's nearest neighbour shares its category ` +
    `${Math.round(100 * inModel)}% of the time in the model's ${REAL_SETTINGS.factorCount} dimensions, ` +
    `${(inModel / chance).toFixed(1)} times what guessing would score (${Math.round(100 * chance)}%). ` +
    `On the flat map it is ${Math.round(100 * onMap)}%. So topic does show up in who was shown what, ` +
    'but faintly, and squeezing 16 dimensions onto a flat map loses most of it. Real people are far messier than our simulated crowd.';
  find('realButton').disabled = false;
}

// ---------- you: reactions, evidence, fold-in ----------

/** What the model is told about one reaction, given the signal choice. UNSEEN means "nothing". */
function evidenceFrom(reaction) {
  if (reaction.liked && state.signal !== 'watch') return LIKE_AS_WATCH_RATIO;
  if (state.signal === 'likes') return UNSEEN;
  return reaction.watchRatio === 0 ? PASSED : reaction.watchRatio; // never started = a pass, not a rejection
}

/** Solves your vector again from the evidence so far. */
function refoldUser() {
  state.foldIn = foldInUser(state.itemFactors, state.itemAppeal, state.evidence, state.settings);
  state.myVector = state.foldIn.vector;
}

/** Re-reads every reaction under the current signal choice and solves your vector again. */
function rebuildEvidence() {
  state.evidence.fill(UNSEEN);
  for (const [item, reaction] of state.reactions) {
    state.evidence[item] = evidenceFrom(reaction);
    tasteMap.markShown(item, preferenceOf(state.evidence[item], state.settings) === 1);
  }
  refoldUser();
}

/** Forgets everything you did, as if a brand-new person had opened the feed. */
function forgetPerson() {
  state.isPretendFan = false; // whoever comes next is you again, not the pretend fan
  tasteMap.setUserLabel('you');
  find('runStatus').hidden = true;
  find('reading').textContent = 'Nothing has been sent to the model yet.';
  state.reactions.clear();
  state.shownMarks = new Float32Array(state.videos.length).fill(UNSEEN);
  state.evidence = new Float32Array(state.videos.length).fill(UNSEEN);
  state.recentCategories = [];
  refoldUser(); // with no evidence this gives all zeros: no taste known yet
  tasteMap.forgetUser(mapPointOf(state.myVector));
  renderVectorPanel();
  renderTowerPanel();
}

/** Called once per card. This is the whole of what the model learns about you. */
function recordReaction(video, watchRatio, liked) {
  const reaction = { watchRatio, liked };
  state.reactions.set(video.index, reaction);
  state.shownMarks[video.index] = 1;
  const told = evidenceFrom(reaction);
  state.evidence[video.index] = told;
  refoldUser();

  const readAsYes = preferenceOf(told, state.settings) === 1;
  tasteMap.markShown(video.index, readAsYes);
  tasteMap.moveUser(mapPointOf(state.myVector));
  find('map').setAttribute('aria-label',
    `Taste map of ${state.videos.length} videos. Your dot has moved ${state.reactions.size} times.`);

  if (told === UNSEEN) {
    find('reading').textContent =
      `About "${video.title}" the model was told nothing: it is set to learn from likes only.`;
  } else {
    const what = told === LIKE_AS_WATCH_RATIO ? 'liked'
      : told === PASSED ? 'passed without watching' : `watched ${Math.round(watchRatio * 100)}%`;
    const verdict = told === LIKE_AS_WATCH_RATIO ? 'a firm yes'
      : told === PASSED ? 'a faint no'
        : readAsYes ? 'a yes' : told < 0.2 ? 'a firm no' : 'a no (a yes needs 70%)';
    find('reading').textContent =
      `About "${video.title}", all the model was told is: video ${video.index + 1}, ${what}. ` +
      `It reads that as ${verdict} and moves ${state.isPretendFan ? 'the fan' : 'you'}.`;
  }
  renderVectorPanel();
  renderTowerPanel();
  if (state.reactions.size === NUDGE_AFTER && !find('reveal').checked) find('revealLabel').classList.add('nudge');
}

// ---------- the feed ----------

/** Chooses the next video with whichever rule the exploration panel is set to. */
function chooseNextVideo() {
  if (state.exploreMode === 'thompson') {
    const spread = MAX_SPREAD * state.exploreAmount;
    return pickThompson(state.foldIn, state.itemFactors, state.itemAppeal, state.shownMarks, spread, state.random);
  }
  const scores = scoreItems(state.myVector, state.itemFactors, state.itemAppeal);
  if (state.exploreMode === 'epsilon') {
    return pickEpsilonGreedy(scores, state.shownMarks, MAX_EPSILON * state.exploreAmount, state.random);
  }
  return bestUnseen(scores, state.shownMarks);
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
  const video = state.videos[item];
  state.currentVideo = video;
  state.recentCategories.push(video.category);
  if (state.recentCategories.length > RECENT_COUNT) state.recentCategories.shift();
  feed.show(video, CATEGORY_COLOURS[video.category], state.categories[video.category].icon);
  tasteMap.setTarget(item);
  explainChoice(item);
  renderBubbleMeter();
}

/** Says in one line why this video is on the card: the two parts of its score, and where it ranked. */
function explainChoice(item) {
  const appeal = state.itemAppeal[item];
  const match = dot(state.myVector, state.itemFactors[item]);
  const scores = scoreItems(state.myVector, state.itemFactors, state.itemAppeal);
  let rank = 1;
  for (let other = 0; other < scores.length; other++) {
    if (state.shownMarks[other] === UNSEEN && scores[other] > scores[item]) rank += 1;
  }
  find('why').textContent =
    `Why this video: general appeal ${appeal.toFixed(2)} ${match < 0 ? '−' : '+'} ` +
    `match with ${state.isPretendFan ? 'the fan' : 'you'} ${Math.abs(match).toFixed(2)} = score ${(appeal + match).toFixed(2)}, ` +
    (rank === 1 ? 'the highest of the videos not yet shown.' : `number ${rank} in line. It was picked to explore.`);
}

/**
 * Fast-forwards the feed with a simple pretend viewer: they watch to the end
 * and like every video of one topic, and pass on everything else without
 * watching. The model is still only told watch ratios, likes and passes; the
 * topic is the pretend viewer's secret.
 */
function runAutopilot(likedCategory, swipesLeft) {
  const name = state.categories[likedCategory].name;
  if (swipesLeft === 0 || !state.currentVideo) {
    finishAutopilot(likedCategory);
    return;
  }
  const video = state.currentVideo;
  const isFavourite = video.category === likedCategory;
  recordReaction(video, isFavourite ? 1 : 0, isFavourite);
  dealNextCard();
  find('runText').textContent = `A pretend ${name} fan is swiping: ${AUTOPILOT_SWIPES - swipesLeft + 1} of ${AUTOPILOT_SWIPES}…`;
  setTimeout(() => runAutopilot(likedCategory, swipesLeft - 1), reducedMotion ? 0 : AUTOPILOT_STEP_MS);
}

/** Starts the pretend viewer as a brand-new person, with the map in view so the run can be watched. */
function startAutopilot() {
  startAgain();
  state.isPretendFan = true;
  tasteMap.setUserLabel('fan');
  setAutopilotBusy(true);
  find('runStatus').hidden = false;
  find('backToExplore').hidden = true;
  document.querySelector('.map-panel').scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'start' });
  runAutopilot(Number(find('autopilotTopic').value), AUTOPILOT_SWIPES);
}

/** Reports one finished run next to the map, and keeps it in a short list so two runs can be compared. */
function finishAutopilot(likedCategory) {
  const name = state.categories[likedCategory].name;
  const outside = state.recentCategories.filter((category) => category !== likedCategory).length;
  const setting = state.exploreMode === 'greedy' ? 'Exploration off'
    : state.exploreMode === 'epsilon' ? `Random sometimes (${Math.round(MAX_EPSILON * state.exploreAmount * 100)}%)`
      : `Explore where unsure (strength ${(MAX_SPREAD * state.exploreAmount).toFixed(1)})`;
  const summary = `${setting}: ${outside} of the fan's last ${state.recentCategories.length} videos were outside ${name}.`;
  find('runText').textContent = `Done. ${summary} The dot is now this fan, not you.`;
  find('backToExplore').hidden = false;
  const log = find('autopilotLog');
  log.append(make('li', '', summary));
  while (log.children.length > 4) log.firstElementChild.remove();
  setAutopilotBusy(false);
}

function setAutopilotBusy(busy) {
  for (const id of ['autopilotButton', 'resetButton', 'trainButton']) find(id).disabled = busy;
  // The last card of a run is already on screen; without this its entrance animation would replay once.
  if (!busy) for (const face of document.querySelectorAll('.card-face.entering')) face.classList.remove('entering');
  document.body.classList.toggle('fast-forward', busy); // the styles switch the card's entrance animation off
}

function startAgain() {
  forgetPerson();
  find('reading').textContent = 'Started again as a brand-new person.';
  dealNextCard();
}

// ---------- drawing the panels ----------

function setRevealed(revealed) {
  tasteMap.setRevealed(revealed);
  find('revealLabel').classList.remove('nudge');
  find('revealNote').textContent = revealed
    ? "These colours are the videos' topics. The model was never given them. It grouped the videos using only who watched what."
    : 'Every dot is one video. The bright dot is you. The dashed line points at the video on the card.';
}

/** Shows how varied your recent recommendations were: 0% is a sealed bubble. */
function renderBubbleMeter() {
  const count = state.recentCategories.length;
  const whose = state.isPretendFan ? "the fan's" : 'your';
  if (count < 5) { // too few to judge
    find('bubbleFill').style.width = '0%';
    find('bubbleText').textContent = `Variety of ${whose} recent recommendations: too early to say.`;
    return;
  }
  const mix = categoryMix(state.recentCategories, state.categories.length);
  const percent = Math.round(mix * 100);
  find('bubbleFill').style.width = `${percent}%`;
  const verdict = mix < 0.3 ? 'A bubble: nearly everything is one topic.'
    : mix < 0.65 ? 'A few topics dominate.'
      : 'Many topics are getting through.';
  find('bubbleText').textContent =
    `Variety of ${whose} last ${count} recommendations: ${percent}% (0% is all one topic, 100% an even spread). ${verdict}`;
}

function renderExploreAmount() {
  find('exploreAmountRow').hidden = state.exploreMode === 'greedy';
  find('exploreAmountText').textContent = state.exploreMode === 'epsilon'
    ? `${Math.round(MAX_EPSILON * state.exploreAmount * 100)}% of videos are picked at random`
    : `strength ${(MAX_SPREAD * state.exploreAmount).toFixed(1)}: how far the imagined you may stray from the model's best guess`;
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
  for (const item of [...state.reactions.keys()].slice(-8).reverse()) {
    const ratio = state.evidence[item];
    const row = make('tr');
    if (ratio === UNSEEN) {
      row.append(make('td', '', `video ${item + 1}`), make('td', '', 'not counted'), make('td', '', '–'), make('td', '', '0'));
    } else if (ratio === PASSED) {
      row.append(make('td', '', `video ${item + 1}`), make('td', '', 'passed, not watched'), make('td', '', 'no (0)'),
        make('td', '', state.settings.passConfidence.toFixed(2)));
    } else {
      row.append(
        make('td', '', `video ${item + 1}`),
        make('td', '', ratio === LIKE_AS_WATCH_RATIO ? 'liked (counts as 2.00)' : ratio.toFixed(2)),
        make('td', '', preferenceOf(ratio, state.settings) === 1 ? 'yes (1)' : 'no (0)'),
        make('td', '', confidenceOf(ratio, state.settings).toFixed(1)),
      );
    }
    evidence.append(row);
  }
  find('evidenceEmpty').hidden = state.reactions.size > 0;

  const scores = scoreItems(state.myVector, state.itemFactors, state.itemAppeal);
  const upNext = find('upNext');
  upNext.replaceChildren();
  for (const item of topUnseen(scores, 5)) upNext.append(make('li', '', `video ${item + 1}: score ${scores[item].toFixed(2)}`));
}

function fillTopicPicker() {
  const picker = find('autopilotTopic');
  picker.replaceChildren();
  state.categories.forEach((category, index) => {
    const option = make('option', '', `${category.icon} ${category.name}`);
    option.value = String(index);
    picker.append(option);
  });
}

function make(tag, className = '', text = '') {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

// ---------- connecting the controls ----------

/** Shows a slider's current value in words next to it. */
function describeSliders() {
  find('crowdUsersText').textContent = `${Number(find('crowdUsers').value).toLocaleString('en')} people`;
  find('crowdTopicsText').textContent = `${find('crowdTopics').value} hidden topics`;
  find('crowdNoiseText').textContent = `${find('crowdNoise').value}% noise in each watch time`;
  find('crowdSparsityText').textContent = `each person never saw ${find('crowdSparsity').value}% of the videos`;
}

function wireControls() {
  find('reveal').addEventListener('change', (event) => setRevealed(event.target.checked));

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
  find('autopilotButton').addEventListener('click', startAutopilot);
  find('backToExplore').addEventListener('click', () =>
    find('explorePanel').scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'start' }));
  find('resetButton').addEventListener('click', startAgain);

  for (const radio of document.querySelectorAll('input[name="signal"]')) {
    radio.addEventListener('change', () => {
      state.signal = radio.value;
      rebuildEvidence();
      tasteMap.moveUser(mapPointOf(state.myVector));
      renderVectorPanel();
      renderTowerPanel();
    });
  }
  find('newVideoButton').addEventListener('click', uploadNewVideo);
  find('realButton').addEventListener('click', trainOnRealPeople);

  for (const id of ['crowdUsers', 'crowdTopics', 'crowdNoise', 'crowdSparsity']) {
    find(id).addEventListener('input', describeSliders);
  }
  find('trainButton').addEventListener('click', requestTraining);
  find('labPanel').addEventListener('toggle', () =>
    drawErrorChart(find('errorChart'), state.errorHistory, DEFAULT_SETTINGS.sweepCount, theme));

  describeSliders();
  renderExploreAmount();
}

wireControls();
requestTraining();
