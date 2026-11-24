// Tests for twotower.js. The heart of this file is the numerical gradient
// check: nudge one number, see how much the loss really changes, and compare
// that with what our hand-written backward pass claims.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  TWO_TOWER_DEFAULTS, linearForward, linearBackward, reluForward, reluBackward, softmaxCrossEntropy,
  createTwoTower, zerosLike, arraysOf, batchLossAndGradients, createAdam, adamStep, trainTwoTower,
  embedItem, embedNewItem, embedUser,
} from '../js/twotower.js';
import { simulateCrowd } from '../js/sim.js';
import { DEFAULT_SETTINGS, preferenceOf } from '../js/mf.js';
import { dot, norm } from '../js/linalg.js';
import { makeRandom, gaussian } from '../js/random.js';

const NUDGE = 1e-5;

/** Asserts two gradients agree to about five significant figures. */
function assertClose(measured, claimed, label) {
  const scale = Math.max(1e-7, Math.abs(measured) + Math.abs(claimed));
  assert.ok(Math.abs(measured - claimed) / scale < 1e-5, `${label}: measured ${measured}, backward pass says ${claimed}`);
}

/** How much lossOf() changes per unit change of array[index], measured by nudging it both ways. */
function measuredSlope(array, index, lossOf) {
  const original = array[index];
  array[index] = original + NUDGE;
  const up = lossOf();
  array[index] = original - NUDGE;
  const down = lossOf();
  array[index] = original;
  return (up - down) / (2 * NUDGE);
}

test('linearBackward matches measured slopes for weights, bias and input', () => {
  const random = makeRandom(1);
  const layer = {
    weights: [Float64Array.of(0.3, -0.7, 0.2), Float64Array.of(-0.5, 0.1, 0.9)],
    bias: Float64Array.of(0.05, -0.02),
  };
  const input = Float64Array.of(gaussian(random), gaussian(random), gaussian(random));
  const pull = Float64Array.of(0.8, -1.3); // pretend the loss is pull . output
  const lossOf = () => dot(pull, linearForward(layer, input));
  const layerGradient = { weights: [new Float64Array(3), new Float64Array(3)], bias: new Float64Array(2) };
  const inputGradient = linearBackward(layer, input, pull, layerGradient);
  for (let row = 0; row < 2; row++) {
    assertClose(measuredSlope(layer.bias, row, lossOf), layerGradient.bias[row], `bias ${row}`);
    for (let column = 0; column < 3; column++) {
      assertClose(measuredSlope(layer.weights[row], column, lossOf), layerGradient.weights[row][column], `weight ${row},${column}`);
    }
  }
  for (let column = 0; column < 3; column++) assertClose(measuredSlope(input, column, lossOf), inputGradient[column], `input ${column}`);
});

test('reluForward zeroes negatives and reluBackward blocks their gradient', () => {
  const values = Float64Array.of(-2, 0.5, 3, -0.1);
  assert.deepEqual(Array.from(reluForward(values)), [0, 0.5, 3, 0]);
  const pull = Float64Array.of(1, 2, 3, 4);
  const gradient = reluBackward(values, pull);
  const lossOf = () => dot(pull, reluForward(values));
  for (let index = 0; index < 4; index++) assertClose(measuredSlope(values, index, lossOf), gradient[index], `relu ${index}`);
});

test('softmaxCrossEntropy gives minus log probability and the right gradient', () => {
  const scores = Float64Array.of(1.0, 2.0, 0.5, -1.0);
  const { loss, scoreGradient } = softmaxCrossEntropy(scores, 1);
  const total = Math.exp(1.0) + Math.exp(2.0) + Math.exp(0.5) + Math.exp(-1.0);
  assert.ok(Math.abs(loss + Math.log(Math.exp(2.0) / total)) < 1e-12);
  const lossOf = () => softmaxCrossEntropy(scores, 1).loss;
  let sum = 0;
  for (let index = 0; index < 4; index++) {
    assertClose(measuredSlope(scores, index, lossOf), scoreGradient[index], `score ${index}`);
    sum += scoreGradient[index];
  }
  assert.ok(Math.abs(sum) < 1e-12, 'the gradients over the scores add up to zero');
});

test('softmaxCrossEntropy survives very large scores', () => {
  const { loss } = softmaxCrossEntropy(Float64Array.of(1000, 999), 0);
  assert.ok(Number.isFinite(loss) && loss > 0);
});

function tinyWorld() {
  const random = makeRandom(5);
  const settings = { ...TWO_TOWER_DEFAULTS, embeddingSize: 3, hiddenSize: 5, outputSize: 4 };
  const itemTags = Array.from({ length: 7 }, (_, item) => Float64Array.from({ length: 3 }, (_, tag) => (item % 3 === tag ? 1 : 0)));
  const model = createTwoTower(7, 3, settings, random);
  // Biases start at exactly zero, which puts the person with no history right on
  // ReLU's bend, where a slope cannot be measured. Move them off it for the check.
  for (const layer of [model.itemHidden, model.itemOutput, model.userHidden, model.userOutput]) {
    for (let row = 0; row < layer.bias.length; row++) layer.bias[row] = 0.1 * gaussian(random);
  }
  const batch = [
    { history: [1, 4], item: 0, hideId: false },
    { history: [2], item: 3, hideId: true }, // a hidden ID
    { history: [], item: 5, hideId: false }, // a person with no history
    { history: [0, 3, 6], item: 6, hideId: false },
  ];
  return { model, batch, itemTags };
}

test('the whole model: every group of numbers gets the gradient that nudging it measures', () => {
  const { model, batch, itemTags } = tinyWorld();
  const gradients = zerosLike(model);
  batchLossAndGradients(model, batch, itemTags, gradients);
  const lossOf = () => batchLossAndGradients(model, batch, itemTags, zerosLike(model));
  const values = arraysOf(model);
  const claimed = arraysOf(gradients);
  let checked = 0;
  for (let a = 0; a < values.length; a++) {
    for (let i = 0; i < values[a].length; i++) {
      assertClose(measuredSlope(values[a], i, lossOf), claimed[a][i], `array ${a}, entry ${i}`);
      checked++;
    }
  }
  assert.equal(checked, 145, 'every one of the model\'s 145 numbers should have been checked');
});

test('a hidden ID sends no gradient to that video\'s ID numbers', () => {
  const { model, batch, itemTags } = tinyWorld();
  const gradients = zerosLike(model);
  batchLossAndGradients(model, batch, itemTags, gradients);
  assert.ok(gradients.itemEmbedding[3].every((value) => value === 0), 'video 3 had its ID hidden');
  assert.ok(gradients.itemEmbedding[0].some((value) => value !== 0), 'video 0 did not');
});

test('an Adam step lowers the loss on the batch it was computed from', () => {
  const { model, batch, itemTags } = tinyWorld();
  const adam = createAdam(model);
  const before = batchLossAndGradients(model, batch, itemTags, zerosLike(model));
  for (let step = 0; step < 20; step++) {
    const gradients = zerosLike(model);
    batchLossAndGradients(model, batch, itemTags, gradients);
    adamStep(model, gradients, adam, 0.01);
  }
  const after = batchLossAndGradients(model, batch, itemTags, zerosLike(model));
  assert.ok(after < before, `loss went from ${before} to ${after}`);
});

test('trained on a simulated crowd, it places a brand-new video among videos with the same tag', () => {
  const topicCount = 5;
  const crowd = simulateCrowd({ userCount: 150, itemCount: 120, topicCount, noise: 0.15, sparsity: 0, seed: 5 });
  const likedByUser = crowd.watchRatios.map((row) => {
    const liked = [];
    row.forEach((ratio, item) => { if (preferenceOf(ratio, DEFAULT_SETTINGS) === 1) liked.push(item); });
    return liked;
  });
  const tagOf = (topic) => Float64Array.from({ length: topicCount }, (_, tag) => (tag === topic ? 1 : 0));
  const itemTags = Array.from(crowd.itemTopics, tagOf);
  const settings = { ...TWO_TOWER_DEFAULTS, stepCount: 300 };
  const random = makeRandom(8);
  const model = createTwoTower(120, topicCount, settings, random);
  const lossHistory = trainTwoTower(model, likedByUser, itemTags, settings, random);

  const average = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
  // In this small crowd each person likes over half the videos, so most "wrong"
  // videos in a batch are not really wrong and the loss cannot fall very far.
  const fall = average(lossHistory.slice(0, 20)) - average(lossHistory.slice(-20));
  assert.ok(fall > 0.1, `training should lower the loss; it fell by ${fall}`);

  const known = Array.from({ length: 120 }, (_, item) => embedItem(model, item, itemTags[item]));
  const cosine = (a, b) => dot(a, b) / (norm(a) * norm(b));
  let sameTopic = 0;
  for (let topic = 0; topic < topicCount; topic++) {
    const fresh = embedNewItem(model, tagOf(topic)); // never seen in training, no ID numbers at all
    let nearest = 0;
    for (let item = 1; item < 120; item++) if (cosine(fresh, known[item]) > cosine(fresh, known[nearest])) nearest = item;
    if (crowd.itemTopics[nearest] === topic) sameTopic++;
  }
  assert.equal(sameTopic, topicCount, 'every brand-new video should land beside videos that share its tag');

  // A person with no history still gets a vector, and a fan's top videos are their topic.
  assert.equal(embedUser(model, []).length, settings.outputSize);
  const fan = embedUser(model, [0, 5, 10]); // three videos of topic 0
  const ranked = known.map((vector, item) => [dot(fan, vector), item]).sort((a, b) => b[0] - a[0]).slice(0, 10);
  const hits = ranked.filter(([, item]) => crowd.itemTopics[item] === 0 || crowd.itemTopics[item] === 1 || crowd.itemTopics[item] === 4).length;
  assert.ok(hits >= 8, `only ${hits} of a topic-0 fan's top 10 were topic 0 or its ring neighbours`);
});
