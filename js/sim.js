// sim.js: a made-up crowd whose hidden tastes we know.
//
// Real data cannot tell us whether the model found the "right" structure,
// because nobody knows the right answer. Here we plant the answer: every
// video secretly belongs to a topic and every person secretly has favourite
// topics. We only hand the model the resulting grid of watch ratios. If the
// model's map groups videos by the topics it was never shown, it works.

import { makeRandom, gaussian } from './random.js';
import { UNSEEN } from './mf.js';
import { dot, norm } from './linalg.js';

/**
 * Topic names for the simulated crowd, in ring order: each topic shares fans
 * with the ones next to it, and the last wraps round to the first.
 * They label the reveal; the model never sees them.
 */
export const TOPIC_NAMES = ['Dance', 'Music', 'Comedy', 'Gaming', 'Football', 'Travel', 'Cooking', 'DIY'];

/**
 * How quickly a person's enjoyment falls off as a topic gets further round
 * the ring from where they stand: the width of a bell curve, in topic steps.
 * It is not a cut-off. At 1.2, a person ends up saying yes to about three
 * neighbouring topics.
 */
export const TASTE_WIDTH = 1.2;

/** The longest watch we record, as a multiple of the video's length. */
export const MAX_WATCH_RATIO = 3;

/**
 * Builds a simulated crowd.
 *   userCount, itemCount, topicCount   sizes
 *   noise     how much random wobble is added to each watch ratio
 *   sparsity  share of cells where the person never saw the video (0 to 1)
 * Returns the watch-ratio grid plus the hidden truth (each video's topic and
 * each person's taste for every topic) so that we can mark the model's work.
 */
export function simulateCrowd({ userCount, itemCount, topicCount, noise, sparsity, seed, tasteWidth = TASTE_WIDTH }) {
  const random = makeRandom(seed);

  const itemTopics = new Int32Array(itemCount);
  const itemAppeal = new Float64Array(itemCount); // some videos are better made than others
  for (let item = 0; item < itemCount; item++) {
    itemTopics[item] = item % topicCount;
    itemAppeal[item] = 0.1 * gaussian(random);
  }

  const userTastes = new Array(userCount);
  for (let user = 0; user < userCount; user++) userTastes[user] = drawTaste(topicCount, tasteWidth, random);

  const watchRatios = new Array(userCount);
  for (let user = 0; user < userCount; user++) {
    watchRatios[user] = new Float32Array(itemCount);
    for (let item = 0; item < itemCount; item++) {
      if (random() < sparsity) {
        watchRatios[user][item] = UNSEEN;
        continue;
      }
      const expected = 0.15 + 1.1 * userTastes[user][itemTopics[item]] + itemAppeal[item];
      const observed = expected + noise * gaussian(random);
      watchRatios[user][item] = Math.min(MAX_WATCH_RATIO, Math.max(0, observed));
    }
  }
  return { watchRatios, itemTopics, userTastes, topicCount };
}

/**
 * One person's hidden taste. Picture the topics spaced round a ring. Each
 * person stands somewhere on that ring and enjoys a topic less the further
 * away it is (a bell-shaped fall-off), so neighbouring topics share fans.
 */
function drawTaste(topicCount, tasteWidth, random) {
  const home = random() * topicCount; // where this person stands on the ring
  const taste = new Float64Array(topicCount);
  for (let topic = 0; topic < topicCount; topic++) {
    const direct = Math.abs(topic - home);
    const steps = Math.min(direct, topicCount - direct); // go round whichever way is shorter
    taste[topic] = Math.exp(-(steps * steps) / (2 * tasteWidth * tasteWidth));
  }
  return taste;
}

/**
 * Marks the model's work against the hidden truth.
 * For each video, find the video whose learned vector points most nearly the
 * same way (largest cosine). Count how often that nearest neighbour shares
 * the hidden topic. 1.0 = always; pure chance = about 1 / number of topics.
 */
export function neighbourAgreement(itemFactors, itemTopics) {
  const lengths = itemFactors.map((factor) => norm(factor) || 1);
  let agreeing = 0;
  for (let item = 0; item < itemFactors.length; item++) {
    let nearest = -1;
    let nearestCosine = -Infinity;
    for (let other = 0; other < itemFactors.length; other++) {
      if (other === item) continue;
      const cosine = dot(itemFactors[item], itemFactors[other]) / (lengths[item] * lengths[other]);
      if (cosine > nearestCosine) {
        nearestCosine = cosine;
        nearest = other;
      }
    }
    if (itemTopics[nearest] === itemTopics[item]) agreeing++;
  }
  return agreeing / itemFactors.length;
}
