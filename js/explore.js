// explore.js: three ways to choose the next video, and a meter for the bubble.
//
// Always showing the top-scoring video ("greedy") looks sensible, but the feed
// then only ever learns about the corner of the map you are already in. That
// is how a filter bubble forms. Exploring means sometimes showing a video the
// model is NOT sure you will like, to find out.

import { backSubstituteTranspose } from './linalg.js';
import { gaussian } from './random.js';
import { UNSEEN, scoreItems, bestUnseen } from './mf.js';

/**
 * Epsilon-greedy: with probability epsilon ignore the scores and show a
 * random unseen video; otherwise show the best one. Simple, but blind: the
 * random pick is as likely to be something you obviously dislike as not.
 */
export function pickEpsilonGreedy(scores, myWatchRatios, epsilon, random) {
  if (random() >= epsilon) return bestUnseen(scores, myWatchRatios);
  const unseen = [];
  for (let item = 0; item < scores.length; item++) if (myWatchRatios[item] === UNSEEN) unseen.push(item);
  if (unseen.length === 0) return -1;
  return unseen[Math.floor(random() * unseen.length)];
}

/**
 * Draws one plausible version of your vector.
 * The fold-in gives a best guess (vector) and a matrix that says how firmly
 * the evidence pins each direction down (precision = L L-transpose, with L =
 * lower). The uncertainty is the inverse of that matrix. To draw from it we
 * take plain bell-curve noise z and solve L-transpose x = z: the result has
 * exactly that inverse as its spread. "spread" scales the noise; 0 = none.
 */
export function sampleUserVector(vector, lower, spread, random) {
  const noise = new Float64Array(vector.length);
  for (let d = 0; d < noise.length; d++) noise[d] = gaussian(random);
  const wobble = backSubstituteTranspose(lower, noise);
  const sampled = new Float64Array(vector.length);
  for (let d = 0; d < sampled.length; d++) sampled[d] = vector[d] + spread * wobble[d];
  return sampled;
}

/**
 * Thompson sampling: instead of scoring with your best-guess vector, draw one
 * plausible vector (see above) and show the best video for THAT person.
 * When the model knows little about you the draws vary widely, so the feed
 * explores; as evidence builds up the draws agree and it settles down. It
 * explores where it is unsure, not at random.
 */
export function pickThompson(foldIn, itemFactors, itemAppeal, myWatchRatios, spread, random) {
  const sampled = sampleUserVector(foldIn.vector, foldIn.lower, spread, random);
  return bestUnseen(scoreItems(sampled, itemFactors, itemAppeal), myWatchRatios);
}

/**
 * The bubble meter. Looks at the categories of the last few recommendations
 * and returns their entropy, scaled so that 0 = every one was the same topic
 * (a sealed bubble) and 1 = spread evenly over all topics.
 * Entropy = minus the sum of share x log(share) over the topics.
 */
export function categoryMix(recentCategories, categoryCount) {
  if (recentCategories.length === 0 || categoryCount < 2) return 0;
  const counts = new Float64Array(categoryCount);
  for (const category of recentCategories) counts[category]++;
  let entropy = 0;
  for (const count of counts) {
    if (count === 0) continue;
    const share = count / recentCategories.length;
    entropy -= share * Math.log(share);
  }
  return entropy / Math.log(categoryCount);
}
