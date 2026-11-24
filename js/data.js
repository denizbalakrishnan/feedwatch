// data.js: what a human sees for each video.
//
// Everything on a card (title, icon, length) is made up here from the
// video's hidden topic. None of it is ever passed to the model, which
// receives the watch-ratio grid alone.

import { TOPIC_NAMES } from './sim.js';

/** The crowd the page opens with. The lab's sliders change these. */
export const DEFAULT_CROWD = { userCount: 300, itemCount: 480, topicCount: 8, noise: 0.25, sparsity: 0, seed: 2 };

const TOPIC_ICONS = ['💃', '🎵', '😂', '🎮', '⚽', '✈️', '🍳', '🔨'];

const TITLE_STEMS = [
  ['Street dance', 'Dance challenge', 'Studio routine', 'Slow-mo spin'],
  ['Guitar cover', 'Drum fill', 'Piano loop', 'New beat'],
  ['Prank fail', 'Stand-up bit', 'Office sketch', 'Dad joke'],
  ['Speedrun', 'Boss fight', 'Clutch round', 'New build'],
  ['Top-corner goal', 'Skill move', 'Match day', 'Keeper save'],
  ['Night market', 'Mountain trail', 'Hidden beach', 'City in a day'],
  ['One-pan dinner', 'Street food', 'Five-minute snack', 'Noodle pull'],
  ['Desk makeover', 'Quick fix', 'Shelf build', 'Paint flip'],
];

/**
 * Fetches the real-data file (made by scripts/build_data.py from KuaiRec) and
 * unpacks it. The file stores one bit per (person, video): 1 if the real app
 * showed that person that video. Returns the grid as rows of 1s and 0s, the
 * category names, and each video's category (kept back for marking only).
 */
export async function loadRealCrowd() {
  const file = await (await fetch('data/kuairec_shown.json')).json();
  const bytes = Uint8Array.from(atob(file.shownBits), (character) => character.charCodeAt(0));
  const grid = [];
  for (let user = 0; user < file.userCount; user++) {
    const row = new Float32Array(file.videoCount);
    for (let video = 0; video < file.videoCount; video++) {
      const position = user * file.videoCount + video;
      row[video] = (bytes[position >> 3] >> (position & 7)) & 1; // pick one bit out of its byte
    }
    grid.push(row);
  }
  return { grid, categories: file.categories, videoCategory: file.videoCategory };
}

/**
 * Turns the simulator's hidden topics into things a person can look at:
 *   videos      for each video a made-up title, a length in seconds and its
 *               hidden category (used only by the reveal)
 *   categories  a name and an icon for each hidden category
 */
export function describeVideos(itemTopics, topicCount) {
  const categories = [];
  for (let topic = 0; topic < topicCount; topic++) {
    categories.push({ name: TOPIC_NAMES[topic], icon: TOPIC_ICONS[topic] });
  }
  const videos = [];
  for (let item = 0; item < itemTopics.length; item++) {
    const category = itemTopics[item];
    const stems = TITLE_STEMS[category];
    const round = Math.floor(item / topicCount);
    videos.push({
      index: item,
      category,
      title: `${stems[round % stems.length]} #${item + 1}`,
      seconds: 3 + (round % 3),
    });
  }
  return { videos, categories };
}
