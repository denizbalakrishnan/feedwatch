// mf.js: the mechanism. Matrix factorisation for implicit feedback, trained
// with alternating least squares (ALS).
//
// The idea of turning each observation into a preference and a confidence,
// and training by ALS, comes from Hu, Koren & Volinsky (2008), "Collaborative
// Filtering for Implicit Feedback Datasets". Our formulas for the two are our
// own (see preferenceOf and confidenceOf), because we have watch ratios for
// videos people were shown, where the paper had counts and no "not shown".
// The per-video "appeal" number (a bias term) is from Koren, Bell & Volinsky
// (2009), "Matrix Factorization Techniques for Recommender Systems". That
// paper also has an overall average and a per-person bias; we keep only the
// per-video one, because adding the same constant to all of one person's
// scores cannot change the order of that person's feed.
//
// The only input is a grid of watch ratios: one row per person, one column
// per video, each cell = time watched / video length. The model never sees a
// title, a picture or a topic. It learns
//   - for every video, one "appeal" number: how much people in general watch it
//   - for every person and every video, a short list of numbers (a "vector")
// so that   appeal + person . video   is close to 1 when that person would
// watch that video and close to 0 when they would not.
//
// Notation used in the comments:
//   k      number of entries in each vector (factorCount; the papers call it f)
//   p      preference: 1 = "watched enough to count as a yes", 0 = "no"
//   c      confidence: how much we trust that p
//   b      a video's appeal (its bias)
//   lambda regularisation: a pull towards zero that stops wild vectors
//   alpha  how fast confidence grows with evidence (confidenceGain)

import { zeroMatrix, dot, addScaled, addOuterProduct, choleskyFactor, choleskySolve } from './linalg.js';
import { makeRandom, gaussian } from './random.js';

/** Marks a cell where the person was never shown the video. */
export const UNSEEN = -1;

/**
 * Marks a video the person was shown but moved past without playing. In our
 * feed a video only plays when tapped, so a pass says much less than
 * starting a video and giving up on it: it is a hint, not a verdict.
 */
export const PASSED = -2;

/** The numbers a user can change in the lab. See LEARN/decisions.md for why each default. */
export const DEFAULT_SETTINGS = {
  factorCount: 8, // k
  regularisation: 2, // lambda
  confidenceGain: 10, // alpha
  likeThreshold: 0.7, // watch ratio at or above this counts as a yes
  passConfidence: 0.25, // how much a pass (see PASSED) counts: a quarter of the weakest watched "no"
  sweepCount: 12,
  seed: 1,
};

/**
 * A tap on the heart is counted as a watch ratio of 2.0. The KuaiRec authors
 * suggest calling a watch ratio above 2.0 a "like"; we use their number the
 * other way round, to turn a like into a watch ratio.
 */
export const LIKE_AS_WATCH_RATIO = 2.0;

/** Preference p: 1 if the person watched at least the threshold share of the video, else 0. */
export function preferenceOf(watchRatio, settings) {
  return watchRatio >= settings.likeThreshold ? 1 : 0;
}

/**
 * Confidence c: how strongly one cell should count.
 * A video the person was never shown tells us nothing, so it counts zero.
 * (The 2008 paper counts it as a weak "no" because its data could not tell
 * "not shown" from "not wanted". A feed knows what it showed, and so do we.)
 * For a video they did see, confidence grows with how far the watch ratio is
 * from the threshold, in either direction: an instant skip is a firm no, a
 * re-watch is a firm yes, and a watch that stops right at the threshold is a
 * coin-flip, so it counts least. This formula is ours, not the paper's: the
 * paper has c = 1 + alpha x count, which has no way to express a firm "no".
 */
export function confidenceOf(watchRatio, settings) {
  if (watchRatio === UNSEEN) return 0;
  if (watchRatio === PASSED) return settings.passConfidence;
  return 1 + settings.confidenceGain * Math.abs(watchRatio - settings.likeThreshold);
}

/**
 * The one equation the whole model is built from.
 * Holding the other side's vectors v still, the best vector x for one person
 * (or one video) solves a ridge regression:
 *     (sum of c v v-transpose + lambda I) x = sum of c (p - alreadyExplained) v
 * "alreadyExplained" is the part of each answer something else accounts for
 * (a video's appeal, when we are solving for a person). Pass null for none.
 * The matrix on the left is called "precision" here; we build it, factorise
 * it with Cholesky and solve. We also return the Cholesky factor, because
 * the same matrix says how unsure the model is about x (used for exploring).
 */
export function solveFactor(otherFactors, watchRatios, settings, alreadyExplained = null) {
  const size = otherFactors[0].length;
  const precision = zeroMatrix(size, size);
  for (let d = 0; d < size; d++) precision[d][d] = settings.regularisation;
  const rightSide = new Float64Array(size);
  for (let index = 0; index < otherFactors.length; index++) {
    const confidence = confidenceOf(watchRatios[index], settings);
    if (confidence === 0) continue;
    const leftToExplain = preferenceOf(watchRatios[index], settings) - (alreadyExplained ? alreadyExplained[index] : 0);
    addOuterProduct(precision, otherFactors[index], confidence);
    addScaled(rightSide, otherFactors[index], confidence * leftToExplain);
  }
  const lower = choleskyFactor(precision);
  return { vector: choleskySolve(lower, rightSide), lower };
}

/**
 * Places a brand-new person on the map without retraining anything.
 * The video vectors and appeals stay fixed and we solve the same equation
 * the training uses for every other person. With no history the answer is
 * the zero vector: no taste yet, so the feed falls back on appeal alone.
 * This runs after every swipe, which is why your dot can move instantly.
 */
export function foldInUser(itemFactors, itemAppeal, myWatchRatios, settings) {
  return solveFactor(itemFactors, myWatchRatios, settings, itemAppeal);
}

/**
 * Trains the model by alternating least squares.
 * Hold the videos still and solve for each person exactly; then hold the
 * people still and solve for each video exactly; repeat. Each half-step can
 * only lower the loss, so training cannot diverge and needs no learning rate.
 * watchRatios[user][item] is the grid. onSweep, if given, is called after
 * every sweep so the page can draw progress.
 */
export function trainALS(watchRatios, settings, onSweep = null) {
  const userCount = watchRatios.length;
  const itemCount = watchRatios[0].length;
  const size = settings.factorCount;
  const ratiosByItem = flipGrid(watchRatios);
  const random = makeRandom(settings.seed);

  const itemFactors = zeroMatrix(itemCount, size);
  for (const factor of itemFactors) {
    for (let d = 0; d < size; d++) factor[d] = 0.1 * gaussian(random); // small random start
  }
  const itemAppeal = new Float64Array(itemCount);
  const userFactors = zeroMatrix(userCount, size);
  // Each person's vector with a constant 1 added on the end. Solving a video
  // against these gives it one extra number that counts the same for
  // everybody: that extra number is the video's appeal.
  const usersPlusOne = zeroMatrix(userCount, size + 1);
  for (const row of usersPlusOne) row[size] = 1;
  const lossHistory = [];

  for (let sweep = 1; sweep <= settings.sweepCount; sweep++) {
    for (let user = 0; user < userCount; user++) {
      userFactors[user] = solveFactor(itemFactors, watchRatios[user], settings, itemAppeal).vector;
      usersPlusOne[user].set(userFactors[user]);
    }
    for (let item = 0; item < itemCount; item++) {
      const solved = solveFactor(usersPlusOne, ratiosByItem[item], settings).vector;
      itemFactors[item] = solved.slice(0, size);
      itemAppeal[item] = solved[size];
    }
    const loss = trainingLoss(watchRatios, userFactors, itemFactors, itemAppeal, settings);
    lossHistory.push(loss);
    if (onSweep) onSweep({ sweep, loss, userFactors, itemFactors, itemAppeal });
  }
  return { userFactors, itemFactors, itemAppeal, lossHistory };
}

/** Turns a people-by-videos grid into a videos-by-people grid, so each video's column is one array. */
function flipGrid(grid) {
  const flipped = new Array(grid[0].length);
  for (let column = 0; column < flipped.length; column++) {
    flipped[column] = new Float32Array(grid.length);
    for (let row = 0; row < grid.length; row++) flipped[column][row] = grid[row][column];
  }
  return flipped;
}

/**
 * The number ALS is driving down: for every cell a person was shown,
 * confidence x (p - prediction)^2, plus lambda x the squared size of every
 * vector and every appeal.
 */
export function trainingLoss(watchRatios, userFactors, itemFactors, itemAppeal, settings) {
  let loss = 0;
  for (let user = 0; user < userFactors.length; user++) {
    for (let item = 0; item < itemFactors.length; item++) {
      const ratio = watchRatios[user][item];
      if (ratio === UNSEEN) continue;
      const prediction = itemAppeal[item] + dot(userFactors[user], itemFactors[item]);
      const miss = preferenceOf(ratio, settings) - prediction;
      loss += confidenceOf(ratio, settings) * miss * miss;
    }
  }
  for (const factor of userFactors) loss += settings.regularisation * dot(factor, factor);
  for (const factor of itemFactors) loss += settings.regularisation * dot(factor, factor);
  loss += settings.regularisation * dot(itemAppeal, itemAppeal);
  return loss;
}

/**
 * Hides a share of the answers so the model can be marked on cells it never
 * trained on. Returns a copy of the grid with those cells set to UNSEEN, the
 * hidden cells themselves as [user, item, watchRatio], and a sample of about
 * the same size from the cells that were kept, for a fair comparison.
 */
export function holdOutCells(watchRatios, share, random) {
  const training = watchRatios.map((row) => Float32Array.from(row));
  const heldOut = [];
  const trainingSample = [];
  for (let user = 0; user < watchRatios.length; user++) {
    for (let item = 0; item < watchRatios[user].length; item++) {
      const ratio = watchRatios[user][item];
      if (ratio === UNSEEN) continue;
      if (random() < share) {
        heldOut.push([user, item, ratio]);
        training[user][item] = UNSEEN;
      } else if (random() < share) {
        trainingSample.push([user, item, ratio]);
      }
    }
  }
  return { training, heldOut, trainingSample };
}

/**
 * How wrong the model is on a list of cells: the average of
 * (p - prediction)^2. If this keeps falling on cells the model trained on
 * but rises on hidden cells, the model is memorising noise (overfitting).
 */
export function meanSquaredMiss(cells, userFactors, itemFactors, itemAppeal, settings) {
  if (cells.length === 0) return 0;
  let total = 0;
  for (const [user, item, ratio] of cells) {
    const prediction = itemAppeal[item] + dot(userFactors[user], itemFactors[item]);
    const miss = preferenceOf(ratio, settings) - prediction;
    total += miss * miss;
  }
  return total / cells.length;
}

/**
 * A more practical exam than squared error. For each person, rank the videos
 * that were hidden from training by score, take the top k, and count how
 * many of those the person really did watch enough to count as a yes.
 * scoreOf(user, item) is whichever model is being marked, so two different
 * models can sit exactly the same exam. People with fewer than k hidden
 * videos cannot fill a top k and are left out. Returns the share of real
 * yeses (null if nobody could be marked) and how many people were marked.
 */
export function precisionAtTop(heldOut, scoreOf, settings, k) {
  const hiddenByUser = new Map();
  for (const [user, item, ratio] of heldOut) {
    if (!hiddenByUser.has(user)) hiddenByUser.set(user, []);
    hiddenByUser.get(user).push([scoreOf(user, item), preferenceOf(ratio, settings)]);
  }
  let hits = 0;
  let peopleMarked = 0;
  for (const cells of hiddenByUser.values()) {
    if (cells.length < k) continue;
    cells.sort((a, b) => b[0] - a[0]);
    for (let rank = 0; rank < k; rank++) hits += cells[rank][1];
    peopleMarked += 1;
  }
  return { share: peopleMarked === 0 ? null : hits / (peopleMarked * k), peopleMarked };
}

/**
 * Scores every video for one person: the video's appeal plus the dot product
 * of the two vectors. Higher = recommend sooner.
 */
export function scoreItems(userVector, itemFactors, itemAppeal) {
  const scores = new Float64Array(itemFactors.length);
  for (let item = 0; item < itemFactors.length; item++) {
    scores[item] = itemAppeal[item] + dot(userVector, itemFactors[item]);
  }
  return scores;
}

/** Picks the highest-scoring video the person has not been shown yet, or -1 if none are left. */
export function bestUnseen(scores, myWatchRatios) {
  let best = -1;
  for (let item = 0; item < scores.length; item++) {
    if (myWatchRatios[item] !== UNSEEN) continue;
    if (best === -1 || scores[item] > scores[best]) best = item;
  }
  return best;
}
