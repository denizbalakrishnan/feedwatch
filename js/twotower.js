// twotower.js: the modern method. A small "two-tower" neural network, with
// the forward pass, the backward pass (backpropagation) and the Adam
// optimiser all written by hand.
//
// Matrix factorisation gives each video its own free list of numbers, so a
// video nobody has watched yet has no list at all: the feed cannot place it.
// A two-tower model instead COMPUTES a video's list from things known about
// it (here: its tags), and a person's list from the videos they liked. One
// tower for videos, one for people; the score is still a dot product.
// This follows Covington et al. (2016) and Yi et al. (2019).
//
//   video tower:  [the video's own ID numbers, its tags] -> hidden layer -> output
//   person tower: average of the "history numbers" of videos they liked -> hidden layer -> output
//
// Training uses in-batch sampled softmax: take a batch of (person, a video
// they liked) pairs; for each person, their own video is the right answer
// and every other video in the batch is treated as a wrong one.

import { zeroMatrix, dot } from './linalg.js';
import { gaussian } from './random.js';

/** Sizes and training settings. See LEARN/decisions.md for why each default. */
export const TWO_TOWER_DEFAULTS = {
  embeddingSize: 8, // numbers in a video's ID list and in its history list
  hiddenSize: 16, // units in each tower's hidden layer
  outputSize: 8, // numbers in the final video and person vectors
  batchSize: 64,
  stepCount: 600,
  learningRate: 0.01,
  historyLength: 10, // at most this many liked videos describe a person
  hideIdRate: 0.25, // how often a video's own ID is hidden in training (see trainTwoTower)
  hideHistoryRate: 0.1, // how often the person's whole history is hidden in training
};

// ---------- the building blocks, each with its backward pass ----------

/** A fully connected layer: output = weights x input + bias. */
export function linearForward(layer, input) {
  const output = new Float64Array(layer.bias.length);
  for (let row = 0; row < output.length; row++) output[row] = layer.bias[row] + dot(layer.weights[row], input);
  return output;
}

/**
 * Backward pass of a fully connected layer. Given how much the loss changes
 * with each output (outputGradient), it adds into layerGradient how much the
 * loss changes with each weight and bias, and returns how much it changes
 * with each input, so the layer before can carry on.
 */
export function linearBackward(layer, input, outputGradient, layerGradient) {
  const inputGradient = new Float64Array(input.length);
  for (let row = 0; row < outputGradient.length; row++) {
    const change = outputGradient[row];
    layerGradient.bias[row] += change;
    for (let column = 0; column < input.length; column++) {
      layerGradient.weights[row][column] += change * input[column];
      inputGradient[column] += change * layer.weights[row][column];
    }
  }
  return inputGradient;
}

/** ReLU: negative numbers become zero. This bend is what lets the network learn more than straight lines. */
export function reluForward(values) {
  return values.map((value) => (value > 0 ? value : 0));
}

/** Backward pass of ReLU: the gradient passes through where the input was positive and is blocked elsewhere. */
export function reluBackward(values, outputGradient) {
  return outputGradient.map((change, index) => (values[index] > 0 ? change : 0));
}

/**
 * Softmax with cross-entropy loss. Softmax turns a row of scores into
 * probabilities that add up to 1; the loss is minus the log of the
 * probability given to the right answer. Returns the loss and its gradient
 * with respect to each score, which is simply (probability - 1) for the
 * right answer and (probability) for every wrong one.
 */
export function softmaxCrossEntropy(scores, correctIndex) {
  let highest = -Infinity;
  for (const score of scores) highest = Math.max(highest, score);
  let total = 0;
  const probabilities = new Float64Array(scores.length);
  for (let index = 0; index < scores.length; index++) {
    probabilities[index] = Math.exp(scores[index] - highest); // subtracting the highest avoids overflow
    total += probabilities[index];
  }
  for (let index = 0; index < scores.length; index++) probabilities[index] /= total;
  const loss = -Math.log(probabilities[correctIndex]);
  probabilities[correctIndex] -= 1;
  return { loss, scoreGradient: probabilities };
}

// ---------- the model ----------

function randomMatrix(rowCount, columnCount, scale, random) {
  const matrix = zeroMatrix(rowCount, columnCount);
  for (const row of matrix) for (let column = 0; column < columnCount; column++) row[column] = scale * gaussian(random);
  return matrix;
}

function newLayer(outputSize, inputSize, random) {
  // "He" starting scale: keeps signals a sensible size through a ReLU layer.
  return { weights: randomMatrix(outputSize, inputSize, Math.sqrt(2 / inputSize), random), bias: new Float64Array(outputSize) };
}

/** Creates a two-tower model with small random starting numbers. */
export function createTwoTower(itemCount, tagCount, settings, random) {
  const { embeddingSize, hiddenSize, outputSize } = settings;
  return {
    itemEmbedding: randomMatrix(itemCount, embeddingSize, 0.1, random), // each video's own ID numbers
    historyEmbedding: randomMatrix(itemCount, embeddingSize, 0.1, random), // what liking a video says about a person
    itemHidden: newLayer(hiddenSize, embeddingSize + tagCount, random),
    itemOutput: newLayer(outputSize, hiddenSize, random),
    userHidden: newLayer(hiddenSize, embeddingSize, random),
    userOutput: newLayer(outputSize, hiddenSize, random),
  };
}

/** A copy of the model's shape filled with zeros, used to collect gradients and Adam's running averages. */
export function zerosLike(model) {
  const zeroLayer = (layer) => ({ weights: layer.weights.map((row) => new Float64Array(row.length)), bias: new Float64Array(layer.bias.length) });
  return {
    itemEmbedding: model.itemEmbedding.map((row) => new Float64Array(row.length)),
    historyEmbedding: model.historyEmbedding.map((row) => new Float64Array(row.length)),
    itemHidden: zeroLayer(model.itemHidden),
    itemOutput: zeroLayer(model.itemOutput),
    userHidden: zeroLayer(model.userHidden),
    userOutput: zeroLayer(model.userOutput),
  };
}

/** Every array of numbers in the model, in a fixed order, so the optimiser can walk over them all. */
export function arraysOf(model) {
  const arrays = [...model.itemEmbedding, ...model.historyEmbedding];
  for (const layer of [model.itemHidden, model.itemOutput, model.userHidden, model.userOutput]) {
    arrays.push(...layer.weights, layer.bias);
  }
  return arrays;
}

/**
 * The video tower. idVector is the video's own ID numbers (all zeros for a
 * brand-new video nobody has watched); tagVector has a 1 for each tag the
 * video carries. Returns every intermediate result, which the backward pass needs.
 */
export function itemTowerForward(model, idVector, tagVector) {
  const input = new Float64Array(idVector.length + tagVector.length);
  input.set(idVector);
  input.set(tagVector, idVector.length);
  const beforeRelu = linearForward(model.itemHidden, input);
  const hidden = reluForward(beforeRelu);
  return { input, beforeRelu, hidden, output: linearForward(model.itemOutput, hidden) };
}

/** The person tower. history is a list of video indexes the person liked (it may be empty). */
export function userTowerForward(model, history) {
  const input = new Float64Array(model.userHidden.weights[0].length);
  for (const item of history) {
    for (let d = 0; d < input.length; d++) input[d] += model.historyEmbedding[item][d] / history.length;
  }
  const beforeRelu = linearForward(model.userHidden, input);
  const hidden = reluForward(beforeRelu);
  return { input, beforeRelu, hidden, output: linearForward(model.userOutput, hidden) };
}

function towerBackward(hiddenLayer, outputLayer, pass, outputGradient, hiddenGradients, outputGradients) {
  const hiddenGradient = linearBackward(outputLayer, pass.hidden, outputGradient, outputGradients);
  const beforeReluGradient = reluBackward(pass.beforeRelu, hiddenGradient);
  return linearBackward(hiddenLayer, pass.input, beforeReluGradient, hiddenGradients);
}

/**
 * One training step's worth of arithmetic: the loss on a batch, and how the
 * loss changes with every number in the model (added into "gradients").
 * Each example is { history, item, hideId }: a person described by videos
 * they liked, one more video they liked, and whether to hide that video's ID.
 * itemTags[item] is the tag vector of each video.
 */
export function batchLossAndGradients(model, batch, itemTags, gradients) {
  const size = batch.length;
  const noId = new Float64Array(model.itemEmbedding[0].length);
  const userPasses = batch.map((example) => userTowerForward(model, example.history));
  const itemPasses = batch.map((example) =>
    itemTowerForward(model, example.hideId ? noId : model.itemEmbedding[example.item], itemTags[example.item]));

  // Score every person in the batch against every video in the batch.
  const userGradients = batch.map(() => new Float64Array(userPasses[0].output.length));
  const itemGradients = batch.map(() => new Float64Array(userPasses[0].output.length));
  let loss = 0;
  for (let person = 0; person < size; person++) {
    const scores = new Float64Array(size);
    for (let video = 0; video < size; video++) scores[video] = dot(userPasses[person].output, itemPasses[video].output);
    const result = softmaxCrossEntropy(scores, person); // the right answer for person i is video i
    loss += result.loss / size;
    for (let video = 0; video < size; video++) {
      const change = result.scoreGradient[video] / size;
      for (let d = 0; d < userGradients[person].length; d++) {
        userGradients[person][d] += change * itemPasses[video].output[d];
        itemGradients[video][d] += change * userPasses[person].output[d];
      }
    }
  }

  // Send the gradients back down both towers and into the embedding tables.
  const idSize = noId.length;
  batch.forEach((example, index) => {
    const userInputGradient = towerBackward(model.userHidden, model.userOutput, userPasses[index], userGradients[index],
      gradients.userHidden, gradients.userOutput);
    for (const item of example.history) {
      for (let d = 0; d < idSize; d++) gradients.historyEmbedding[item][d] += userInputGradient[d] / example.history.length;
    }
    const itemInputGradient = towerBackward(model.itemHidden, model.itemOutput, itemPasses[index], itemGradients[index],
      gradients.itemHidden, gradients.itemOutput);
    if (!example.hideId) {
      for (let d = 0; d < idSize; d++) gradients.itemEmbedding[example.item][d] += itemInputGradient[d];
    }
  });
  return loss;
}

// ---------- the optimiser ----------

/** Adam keeps two running averages per number: of its gradient and of its gradient squared. */
export function createAdam(model) {
  return { stepsTaken: 0, average: arraysOf(zerosLike(model)), averageSquare: arraysOf(zerosLike(model)) };
}

/**
 * One Adam update (Kingma & Ba, 2015). Each number moves against its
 * gradient, but the step is scaled by how large that number's gradients
 * usually are, so every number learns at a similar pace. The two "corrected"
 * values undo the fact that the running averages start at zero.
 */
export function adamStep(model, gradients, adam, learningRate) {
  const DECAY = 0.9, DECAY_SQUARE = 0.999, TINY = 1e-8;
  adam.stepsTaken += 1;
  const values = arraysOf(model);
  const changes = arraysOf(gradients);
  for (let a = 0; a < values.length; a++) {
    const average = adam.average[a];
    const averageSquare = adam.averageSquare[a];
    for (let i = 0; i < values[a].length; i++) {
      average[i] = DECAY * average[i] + (1 - DECAY) * changes[a][i];
      averageSquare[i] = DECAY_SQUARE * averageSquare[i] + (1 - DECAY_SQUARE) * changes[a][i] * changes[a][i];
      const corrected = average[i] / (1 - Math.pow(DECAY, adam.stepsTaken));
      const correctedSquare = averageSquare[i] / (1 - Math.pow(DECAY_SQUARE, adam.stepsTaken));
      values[a][i] -= learningRate * corrected / (Math.sqrt(correctedSquare) + TINY);
    }
  }
}

// ---------- training and use ----------

/**
 * Trains the model. likedByUser[user] lists the videos that person liked.
 * Each step draws a batch: a random person with at least two likes, one of
 * their likes as the answer, and up to historyLength of their other likes to
 * describe them. A quarter of the time the answer video's own ID is hidden,
 * which forces the video tower to manage on tags alone; that is exactly the
 * situation of a brand-new video. One time in ten the person's history is
 * hidden instead, so the model also learns what to offer somebody it knows
 * nothing about. onStep(step, loss) reports progress.
 */
export function trainTwoTower(model, likedByUser, itemTags, settings, random, onStep = null) {
  const eligible = [];
  likedByUser.forEach((liked, user) => { if (liked.length >= 2) eligible.push(user); });
  const adam = createAdam(model);
  const lossHistory = [];
  if (eligible.length === 0) return lossHistory; // nobody liked two videos, so there is nothing to learn from
  for (let step = 1; step <= settings.stepCount; step++) {
    const batch = [];
    for (let b = 0; b < settings.batchSize; b++) {
      const liked = likedByUser[eligible[Math.floor(random() * eligible.length)]];
      const item = liked[Math.floor(random() * liked.length)];
      const history = [];
      const triesAllowed = random() < settings.hideHistoryRate ? 0 : settings.historyLength;
      for (let tries = 0; tries < triesAllowed; tries++) {
        const other = liked[Math.floor(random() * liked.length)];
        if (other !== item) history.push(other);
      }
      batch.push({ history, item, hideId: random() < settings.hideIdRate });
    }
    const gradients = zerosLike(model);
    const loss = batchLossAndGradients(model, batch, itemTags, gradients);
    adamStep(model, gradients, adam, settings.learningRate);
    lossHistory.push(loss);
    if (onStep) onStep(step, loss);
  }
  return lossHistory;
}

/** Turns a list of tag numbers into the tag vector the video tower reads: a 1 for each tag carried, 0 elsewhere. */
export function tagsToVector(tags, tagCount) {
  const vector = new Float64Array(tagCount);
  for (const tag of tags) vector[tag] = 1;
  return vector;
}

/** The final vector of a video the model was trained on. */
export function embedItem(model, item, tagVector) {
  return itemTowerForward(model, model.itemEmbedding[item], tagVector).output;
}

/** The final vector of a brand-new video, from its tags alone. Matrix factorisation has no way to do this. */
export function embedNewItem(model, tagVector) {
  return itemTowerForward(model, new Float64Array(model.itemEmbedding[0].length), tagVector).output;
}

/** The final vector of a person, from the videos they liked. An empty history still gives an answer. */
export function embedUser(model, history) {
  return userTowerForward(model, history).output;
}
