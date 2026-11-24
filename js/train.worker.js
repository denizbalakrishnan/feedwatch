// train.worker.js: builds the crowd and trains both models away from the page.
//
// A Web Worker is a second thread. Training here means the page stays smooth
// while the sums are done, even for a crowd of 2,000 people. The worker is
// told the settings and posts back:
//   'sweep'  after every training sweep of the first model (for the error chart)
//   'done'   the first model (matrix factorisation), ready to use
//   'tower'  the second model (two-tower), a moment later

import { simulateCrowd } from './sim.js';
import { UNSEEN, trainALS, holdOutCells, meanSquaredMiss, precisionAtTop, preferenceOf } from './mf.js';
import { TWO_TOWER_DEFAULTS, createTwoTower, trainTwoTower, tagsToVector, embedItem, embedUser } from './twotower.js';
import { dot } from './linalg.js';
import { makeRandom } from './random.js';

/** One cell in ten is hidden from training and used only to mark the models. */
const HELD_OUT_SHARE = 0.1;
/** Both models are marked on each person's top 10 hidden videos. */
const TOP_COUNT = 10;

self.onmessage = (event) => {
  if (event.data.realGrid) {
    trainOnRealGrid(event.data.realGrid, event.data.settings);
    return;
  }
  const { jobId, crowdSettings, modelSettings } = event.data;
  const began = performance.now();

  const crowd = simulateCrowd(crowdSettings);
  const split = holdOutCells(crowd.watchRatios, HELD_OUT_SHARE, makeRandom(crowdSettings.seed + 1000));

  const model = trainALS(split.training, modelSettings, (progress) => {
    const { userFactors, itemFactors, itemAppeal } = progress;
    self.postMessage({
      type: 'sweep',
      jobId,
      sweep: progress.sweep,
      trainingError: meanSquaredMiss(split.trainingSample, userFactors, itemFactors, itemAppeal, modelSettings),
      heldOutError: meanSquaredMiss(split.heldOut, userFactors, itemFactors, itemAppeal, modelSettings),
    });
  });

  let trainedCells = 0;
  for (const row of split.training) for (const ratio of row) if (ratio !== UNSEEN) trainedCells++;

  self.postMessage({
    type: 'done',
    jobId,
    crowdSettings,
    modelSettings,
    seconds: (performance.now() - began) / 1000,
    trainedCells,
    hiddenCells: split.heldOut.length,
    itemTopics: crowd.itemTopics,
    itemFactors: model.itemFactors,
    itemAppeal: model.itemAppeal,
    userFactors: model.userFactors,
  });

  trainSecondModel(jobId, crowd, split, model, modelSettings, crowdSettings.seed);
};

/**
 * The "Real people" panel: trains the very same first model on a grid from a
 * real app, where each cell is 1 if the app showed that person that video
 * and 0 if it did not.
 */
function trainOnRealGrid(grid, settings) {
  const began = performance.now();
  const model = trainALS(grid, settings, (progress) => self.postMessage({ type: 'realSweep', sweep: progress.sweep }));
  self.postMessage({ type: 'real', seconds: (performance.now() - began) / 1000, itemFactors: model.itemFactors });
}

/**
 * Trains the two-tower model on the same training cells, then marks both
 * models on the same hidden cells. The second model is given one thing the
 * first never sees: each video's tag (here, its topic).
 */
function trainSecondModel(jobId, crowd, split, firstModel, modelSettings, seed) {
  const began = performance.now();
  const itemTags = Array.from(crowd.itemTopics, (topic) => tagsToVector([topic], crowd.topicCount));
  const likedByUser = split.training.map((row) => {
    const liked = [];
    row.forEach((ratio, item) => { if (preferenceOf(ratio, modelSettings) === 1) liked.push(item); });
    return liked;
  });

  const random = makeRandom(seed + 2000);
  const tower = createTwoTower(itemTags.length, crowd.topicCount, TWO_TOWER_DEFAULTS, random);
  const lossHistory = trainTwoTower(tower, likedByUser, itemTags, TWO_TOWER_DEFAULTS, random);
  const itemVectors = itemTags.map((tags, item) => embedItem(tower, item, tags));
  const userVectors = likedByUser.map((liked) => embedUser(tower, liked));

  let hiddenYes = 0;
  for (const [, , ratio] of split.heldOut) hiddenYes += preferenceOf(ratio, modelSettings);

  self.postMessage({
    type: 'tower',
    jobId,
    seconds: (performance.now() - began) / 1000,
    stepCount: lossHistory.length,
    tower,
    itemVectors,
    userVectors,
    topCount: TOP_COUNT,
    yesShare: split.heldOut.length ? hiddenYes / split.heldOut.length : 0,
    firstPrecision: precisionAtTop(split.heldOut,
      (user, item) => firstModel.itemAppeal[item] + dot(firstModel.userFactors[user], firstModel.itemFactors[item]),
      modelSettings, TOP_COUNT),
    towerPrecision: precisionAtTop(split.heldOut,
      (user, item) => dot(userVectors[user], itemVectors[item]), modelSettings, TOP_COUNT),
  });
}
