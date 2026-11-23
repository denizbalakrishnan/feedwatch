// Tests for sim.js: can the model recover a structure we planted and hid from it?
import test from 'node:test';
import assert from 'node:assert/strict';
import { simulateCrowd, neighbourAgreement } from '../js/sim.js';
import { DEFAULT_SETTINGS, trainALS, UNSEEN } from '../js/mf.js';

const easy = { userCount: 150, itemCount: 120, topicCount: 5, noise: 0.15, sparsity: 0, seed: 5 };

test('the simulator is repeatable for a given seed', () => {
  const first = simulateCrowd(easy);
  const second = simulateCrowd(easy);
  assert.deepEqual(first.watchRatios, second.watchRatios);
});

test('sparsity hides roughly the requested share of cells', () => {
  const crowd = simulateCrowd({ ...easy, sparsity: 0.4 });
  let unseen = 0;
  for (const row of crowd.watchRatios) for (const ratio of row) if (ratio === UNSEEN) unseen++;
  const share = unseen / (easy.userCount * easy.itemCount);
  assert.ok(Math.abs(share - 0.4) < 0.03, `unseen share was ${share}`);
});

test('on easy settings the model groups videos by the topics it was never shown', () => {
  const crowd = simulateCrowd(easy);
  const { itemFactors } = trainALS(crowd.watchRatios, DEFAULT_SETTINGS);
  const agreement = neighbourAgreement(itemFactors, crowd.itemTopics);
  assert.ok(agreement > 0.9, `neighbour agreement was only ${agreement}`);
});

test('the score is near chance when the labels are scrambled, so it can fail', () => {
  const crowd = simulateCrowd(easy);
  const { itemFactors } = trainALS(crowd.watchRatios, DEFAULT_SETTINGS);
  const scrambled = Int32Array.from(crowd.itemTopics, (_, item) => (item * 7 + Math.floor(item / 5)) % easy.topicCount);
  const agreement = neighbourAgreement(itemFactors, scrambled);
  assert.ok(agreement < 0.5, `scrambled labels still scored ${agreement}`);
});
