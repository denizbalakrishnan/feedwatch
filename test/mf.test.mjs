// Tests for mf.js. Run with: node --test
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  UNSEEN, PASSED, DEFAULT_SETTINGS, preferenceOf, confidenceOf, solveFactor, foldInUser, trainALS, trainingLoss,
  holdOutCells, meanSquaredMiss, precisionAtTop, scoreItems, bestUnseen,
} from '../js/mf.js';
import { dot } from '../js/linalg.js';
import { makeRandom, gaussian } from '../js/random.js';

const settings = { ...DEFAULT_SETTINGS, factorCount: 3, sweepCount: 15 };

function randomFactors(count, size, seed) {
  const random = makeRandom(seed);
  return Array.from({ length: count }, () => Float64Array.from({ length: size }, () => gaussian(random)));
}

function randomRatios(count, seed, unseenShare = 0.3) {
  const random = makeRandom(seed);
  return Float32Array.from({ length: count }, () => (random() < unseenShare ? UNSEEN : 2 * random()));
}

function randomAppeal(count, seed) {
  const random = makeRandom(seed);
  return Float64Array.from({ length: count }, () => 0.5 * random());
}

test('preference is a yes only at or above the threshold', () => {
  assert.equal(preferenceOf(0.69, settings), 0);
  assert.equal(preferenceOf(0.7, settings), 1);
  assert.equal(preferenceOf(UNSEEN, settings), 0);
});

test('confidence is 0 for unseen and grows with distance from the threshold', () => {
  assert.equal(confidenceOf(UNSEEN, settings), 0);
  assert.ok(confidenceOf(0.05, settings) > confidenceOf(0.6, settings), 'an instant skip is a firmer no than a near miss');
  assert.ok(confidenceOf(2.0, settings) > confidenceOf(0.8, settings), 'a re-watch is a firmer yes than a bare pass');
});

test('a pass is a no that counts far less than starting a video and giving up on it', () => {
  assert.equal(preferenceOf(PASSED, settings), 0);
  assert.equal(confidenceOf(PASSED, settings), settings.passConfidence);
  assert.ok(confidenceOf(PASSED, settings) < confidenceOf(0.69, settings), 'weaker than the weakest watched no');
  const items = randomFactors(30, 3, 37);
  const appeal = new Float64Array(30).fill(0.6);
  const reactTo7 = (ratio) => { const mine = new Float32Array(30).fill(UNSEEN); mine[7] = ratio; return mine; };
  const afterPass = foldInUser(items, appeal, reactTo7(PASSED), settings).vector;
  const afterQuitting = foldInUser(items, appeal, reactTo7(0.05), settings).vector;
  assert.ok(dot(afterPass, items[7]) < 0, 'a pass still nudges you away from that video');
  assert.ok(dot(afterPass, afterPass) < dot(afterQuitting, afterQuitting) / 4, 'but much less than quitting it does');
});

test('solveFactor satisfies the ridge-regression equation it claims to solve', () => {
  const others = randomFactors(40, 3, 11);
  const ratios = randomRatios(40, 12);
  const explained = randomAppeal(40, 13);
  const { vector } = solveFactor(others, ratios, settings, explained);
  // Rebuild (sum c v v-transpose + lambda I) x and sum c (p - explained) v the slow, obvious way.
  for (let row = 0; row < 3; row++) {
    let left = settings.regularisation * vector[row];
    let right = 0;
    for (let index = 0; index < others.length; index++) {
      const confidence = confidenceOf(ratios[index], settings);
      left += confidence * others[index][row] * dot(others[index], vector);
      right += confidence * (preferenceOf(ratios[index], settings) - explained[index]) * others[index][row];
    }
    assert.ok(Math.abs(left - right) < 1e-9, `row ${row}: ${left} vs ${right}`);
  }
});

test('foldInUser gives the lowest loss: nudging any entry makes it worse', () => {
  const items = randomFactors(30, 3, 21);
  const appeal = randomAppeal(30, 23);
  const mine = randomRatios(30, 22);
  const { vector } = foldInUser(items, appeal, mine, settings);
  const lossFor = (candidate) => trainingLoss([mine], [candidate], items, appeal, settings);
  const best = lossFor(vector);
  for (let d = 0; d < 3; d++) {
    for (const nudge of [-1e-3, 1e-3]) {
      const moved = Float64Array.from(vector);
      moved[d] += nudge;
      assert.ok(lossFor(moved) > best, `moving entry ${d} by ${nudge} should raise the loss`);
    }
  }
});

test('a person with no history folds in at exactly zero', () => {
  const items = randomFactors(30, 3, 31);
  const { vector } = foldInUser(items, randomAppeal(30, 32), new Float32Array(30).fill(UNSEEN), settings);
  for (const entry of vector) assert.equal(entry, 0);
});

test('one like pulls a new person towards that video, one skip pushes them away', () => {
  const items = randomFactors(30, 3, 35);
  const appeal = new Float64Array(30).fill(0.4);
  const liked = new Float32Array(30).fill(UNSEEN);
  liked[7] = 2.0;
  const skipped = new Float32Array(30).fill(UNSEEN);
  skipped[7] = 0.05;
  assert.ok(dot(foldInUser(items, appeal, liked, settings).vector, items[7]) > 0);
  assert.ok(dot(foldInUser(items, appeal, skipped, settings).vector, items[7]) < 0);
});

test('ALS never raises the loss from one sweep to the next', () => {
  const random = makeRandom(41);
  const grid = Array.from({ length: 25 }, () =>
    Float32Array.from({ length: 30 }, () => (random() < 0.2 ? UNSEEN : 1.8 * random())));
  const { lossHistory } = trainALS(grid, settings);
  assert.equal(lossHistory.length, settings.sweepCount);
  for (let sweep = 1; sweep < lossHistory.length; sweep++) {
    assert.ok(lossHistory[sweep] <= lossHistory[sweep - 1] + 1e-9,
      `sweep ${sweep + 1} loss ${lossHistory[sweep]} rose above ${lossHistory[sweep - 1]}`);
  }
  assert.ok(lossHistory.at(-1) < lossHistory[0], 'training should make progress');
});

test('a video everybody watches gets a higher appeal than one nobody watches', () => {
  const random = makeRandom(45);
  const grid = Array.from({ length: 40 }, () => {
    const row = Float32Array.from({ length: 12 }, () => 1.4 * random());
    row[0] = 1.5; // everybody watches video 0 to the end and beyond
    row[1] = 0.05; // everybody skips video 1 at once
    return row;
  });
  const { itemAppeal } = trainALS(grid, settings);
  assert.ok(itemAppeal[0] > 0.8, `appeal of the hit was ${itemAppeal[0]}`);
  assert.ok(itemAppeal[1] < 0.2, `appeal of the flop was ${itemAppeal[1]}`);
});

test('trainALS gives the same answer for the same seed', () => {
  const random = makeRandom(51);
  const grid = Array.from({ length: 10 }, () => Float32Array.from({ length: 12 }, () => 1.5 * random()));
  const first = trainALS(grid, settings);
  const second = trainALS(grid, settings);
  assert.deepEqual(first.itemFactors, second.itemFactors);
  assert.deepEqual(first.itemAppeal, second.itemAppeal);
});

test('holdOutCells hides about the requested share and never trains on a hidden cell', () => {
  const random = makeRandom(61);
  const grid = Array.from({ length: 40 }, () =>
    Float32Array.from({ length: 50 }, () => (random() < 0.25 ? UNSEEN : 1.6 * random())));
  let seen = 0;
  for (const row of grid) for (const ratio of row) if (ratio !== UNSEEN) seen++;
  const { training, heldOut, trainingSample } = holdOutCells(grid, 0.2, makeRandom(62));
  assert.ok(Math.abs(heldOut.length / seen - 0.2) < 0.04, `held out ${heldOut.length} of ${seen}`);
  for (const [user, item, ratio] of heldOut) {
    assert.equal(training[user][item], UNSEEN);
    assert.equal(grid[user][item], ratio, 'the original grid is left alone');
  }
  for (const [user, item, ratio] of trainingSample) assert.equal(training[user][item], ratio);
});

test('meanSquaredMiss is the average squared gap between yes/no and the prediction', () => {
  const users = [Float64Array.of(1, 0)];
  const items = [Float64Array.of(0.5, 0), Float64Array.of(0, 9)];
  const appeal = Float64Array.of(0.25, 0.5);
  // cell 1: watched 1.0 (a yes, p = 1), predicted 0.25 + 0.5 = 0.75, miss 0.25
  // cell 2: watched 0.1 (a no, p = 0), predicted 0.5 + 0 = 0.5, miss -0.5
  const miss = meanSquaredMiss([[0, 0, 1.0], [0, 1, 0.1]], users, items, appeal, settings);
  assert.ok(Math.abs(miss - (0.0625 + 0.25) / 2) < 1e-12);
});

test('precisionAtTop counts how many of each person\'s top hidden videos were real yeses', () => {
  // Person 0 has four hidden videos; a perfect scorer puts the two yeses first.
  // Person 1 has only one hidden video, too few for a top 2, so is left out.
  const heldOut = [[0, 10, 1.5], [0, 11, 0.1], [0, 12, 0.9], [0, 13, 0.2], [1, 10, 1.0]];
  const perfect = (user, item) => ({ 10: 9, 11: 1, 12: 8, 13: 2 })[item];
  const backwards = (user, item) => -perfect(user, item);
  assert.deepEqual(precisionAtTop(heldOut, perfect, settings, 2), { share: 1, peopleMarked: 1 });
  assert.deepEqual(precisionAtTop(heldOut, backwards, settings, 2), { share: 0, peopleMarked: 1 });
  // Nobody has five hidden videos, so there is no score at all, which is not the same as a score of 0.
  assert.deepEqual(precisionAtTop(heldOut, perfect, settings, 5), { share: null, peopleMarked: 0 });
});

test('scoreItems is appeal plus the dot product with every video', () => {
  const items = [Float64Array.of(1, 0), Float64Array.of(0, 2), Float64Array.of(-1, 1)];
  const appeal = Float64Array.of(0.5, 0, -1);
  assert.deepEqual(Array.from(scoreItems(Float64Array.of(3, 4), items, appeal)), [3.5, 8, 0]);
});

test('bestUnseen skips videos already shown', () => {
  const mine = Float32Array.of(UNSEEN, 0.9, UNSEEN, UNSEEN);
  assert.equal(bestUnseen(Float64Array.of(1, 9, 3, 2), mine), 2, 'item 1 scores highest but was already shown');
  assert.equal(bestUnseen(new Float64Array(2), Float32Array.of(0.1, 0.2)), -1, 'nothing left to show');
});
