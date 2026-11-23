// Tests for explore.js. Run with: node --test
import test from 'node:test';
import assert from 'node:assert/strict';
import { pickEpsilonGreedy, sampleUserVector, pickThompson, categoryMix } from '../js/explore.js';
import { UNSEEN, DEFAULT_SETTINGS, foldInUser, scoreItems, bestUnseen } from '../js/mf.js';
import { makeRandom, gaussian } from '../js/random.js';

const settings = { ...DEFAULT_SETTINGS, factorCount: 2 };

function smallWorld() {
  const random = makeRandom(9);
  const items = Array.from({ length: 40 }, () => Float64Array.of(gaussian(random), gaussian(random)));
  const appeal = Float64Array.from({ length: 40 }, () => 0.3 * random());
  const mine = new Float32Array(40).fill(UNSEEN);
  mine[3] = 2.0;
  mine[8] = 0.05;
  mine[15] = 1.0;
  return { items, appeal, mine, foldIn: foldInUser(items, appeal, mine, settings) };
}

test('epsilon 0 is exactly greedy', () => {
  const { items, appeal, mine, foldIn } = smallWorld();
  const scores = scoreItems(foldIn.vector, items, appeal);
  assert.equal(pickEpsilonGreedy(scores, mine, 0, makeRandom(1)), bestUnseen(scores, mine));
});

test('epsilon 1 picks all over the place but never a video already shown', () => {
  const { items, appeal, mine, foldIn } = smallWorld();
  const scores = scoreItems(foldIn.vector, items, appeal);
  const random = makeRandom(2);
  const picked = new Set();
  for (let trial = 0; trial < 400; trial++) {
    const item = pickEpsilonGreedy(scores, mine, 1, random);
    assert.equal(mine[item], UNSEEN);
    picked.add(item);
  }
  assert.ok(picked.size > 30, `only ${picked.size} different videos were picked`);
});

test('Thompson sampling with no spread is exactly greedy', () => {
  const { items, appeal, mine, foldIn } = smallWorld();
  const greedy = bestUnseen(scoreItems(foldIn.vector, items, appeal), mine);
  for (let seed = 1; seed <= 20; seed++) {
    assert.equal(pickThompson(foldIn, items, appeal, mine, 0, makeRandom(seed)), greedy);
  }
});

test('Thompson sampling homes in on greedy as the spread shrinks', () => {
  const { items, appeal, mine, foldIn } = smallWorld();
  const greedy = bestUnseen(scoreItems(foldIn.vector, items, appeal), mine);
  const agreement = (spread) => {
    const random = makeRandom(5);
    let same = 0;
    for (let trial = 0; trial < 300; trial++) if (pickThompson(foldIn, items, appeal, mine, spread, random) === greedy) same++;
    return same / 300;
  };
  assert.ok(agreement(0.01) > agreement(1), 'less spread should mean more agreement with greedy');
  assert.ok(agreement(0.01) > 0.95);
  assert.ok(agreement(3) < 0.9, 'a wide spread should explore');
});

test('sampled vectors are spread exactly as the inverse of the precision matrix says', () => {
  const { foldIn } = smallWorld();
  // Rebuild precision = L L-transpose and invert the 2x2 by hand.
  const L = foldIn.lower;
  const a = L[0][0] * L[0][0];
  const b = L[0][0] * L[1][0];
  const d = L[1][0] * L[1][0] + L[1][1] * L[1][1];
  const determinant = a * d - b * b;
  const expected = [[d / determinant, -b / determinant], [-b / determinant, a / determinant]];

  const random = makeRandom(11);
  const sums = [[0, 0], [0, 0]];
  const draws = 40000;
  for (let draw = 0; draw < draws; draw++) {
    const sampled = sampleUserVector(foldIn.vector, foldIn.lower, 1, random);
    const offset = [sampled[0] - foldIn.vector[0], sampled[1] - foldIn.vector[1]];
    for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++) sums[r][c] += offset[r] * offset[c];
  }
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 2; c++) {
      const measured = sums[r][c] / draws;
      const tolerance = 0.05 * Math.max(Math.abs(expected[0][0]), Math.abs(expected[1][1]));
      assert.ok(Math.abs(measured - expected[r][c]) < tolerance, `[${r}][${c}] measured ${measured}, expected ${expected[r][c]}`);
    }
  }
});

test('categoryMix is 0 for one topic, 1 for an even spread, and in between otherwise', () => {
  assert.equal(categoryMix([2, 2, 2, 2], 4), 0);
  assert.ok(Math.abs(categoryMix([0, 1, 2, 3], 4) - 1) < 1e-12);
  const lopsided = categoryMix([0, 0, 0, 1], 4);
  assert.ok(lopsided > 0 && lopsided < 1);
  assert.equal(categoryMix([], 4), 0);
});
