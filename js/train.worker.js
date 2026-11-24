// train.worker.js: builds the crowd and trains the model away from the page.
//
// A Web Worker is a second thread. Training here means the page stays smooth
// while the sums are done, even for a crowd of 2,000 people. The worker is
// told the settings, and posts back one message per training sweep (so the
// page can draw the error falling) and one final message with the model.

import { simulateCrowd } from './sim.js';
import { UNSEEN, trainALS, holdOutCells, meanSquaredMiss } from './mf.js';
import { makeRandom } from './random.js';

/** One cell in ten is hidden from training and used only to mark the model. */
const HELD_OUT_SHARE = 0.1;

self.onmessage = (event) => {
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
};
