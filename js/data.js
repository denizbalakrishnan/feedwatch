// data.js: hands the page its crowd.
//
// For now the crowd is simulated (see sim.js). When the real KuaiRec
// subsample is ready, this file will fetch that instead and nothing else has
// to change: the rest of the page relies only on the shape loadCrowd() returns.
//
// Everything a human sees on a card (title, icon) is made up here. None of it
// is ever passed to the model, which receives the watch-ratio grid alone.

import { simulateCrowd, TOPIC_NAMES } from './sim.js';

/** Size and seed of the crowd on the main screen. */
export const MAIN_CROWD = { userCount: 300, itemCount: 480, topicCount: 8, noise: 0.25, sparsity: 0, seed: 2 };

const TOPIC_ICONS = ['💃', '🎵', '😂', '🎮', '⚽', '✈️', '🍳', '🔨', '🐾', '💄'];

const TITLE_STEMS = [
  ['Street dance', 'Dance challenge', 'Studio routine', 'Slow-mo spin'],
  ['Guitar cover', 'Drum fill', 'Piano loop', 'New beat'],
  ['Prank fail', 'Stand-up bit', 'Office sketch', 'Dad joke'],
  ['Speedrun', 'Boss fight', 'Clutch round', 'New build'],
  ['Top-corner goal', 'Skill move', 'Match day', 'Keeper save'],
  ['Night market', 'Mountain trail', 'Hidden beach', 'City in a day'],
  ['One-pan dinner', 'Street food', 'Five-minute snack', 'Noodle pull'],
  ['Desk makeover', 'Quick fix', 'Shelf build', 'Paint flip'],
  ['Puppy zoomies', 'Cat versus box', 'Parrot talk', 'Rescue update'],
  ['Everyday look', 'Glow routine', 'Bold liner', 'Nail art'],
];

/**
 * Returns the crowd the page runs on:
 *   watchRatios  the grid the model is trained on (people x videos)
 *   videos       what a human sees for each video: a made-up title, a length
 *                in seconds and the hidden category used only by the reveal
 *   categories   a name and icon for each hidden category
 */
export function loadCrowd() {
  const crowd = simulateCrowd(MAIN_CROWD);
  const categories = [];
  for (let topic = 0; topic < MAIN_CROWD.topicCount; topic++) {
    categories.push({ name: TOPIC_NAMES[topic], icon: TOPIC_ICONS[topic] });
  }
  const videos = [];
  for (let item = 0; item < MAIN_CROWD.itemCount; item++) {
    const category = crowd.itemTopics[item];
    const stems = TITLE_STEMS[category];
    const round = Math.floor(item / MAIN_CROWD.topicCount);
    videos.push({
      index: item,
      category,
      title: `${stems[round % stems.length]} #${item + 1}`,
      seconds: 3 + (round % 3),
    });
  }
  return {
    sourceLabel: `a simulated crowd of ${MAIN_CROWD.userCount} people and ${MAIN_CROWD.itemCount} videos`,
    watchRatios: crowd.watchRatios,
    videos,
    categories,
  };
}
