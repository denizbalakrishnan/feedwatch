// random.js: repeatable random numbers.
//
// Math.random() gives different numbers on every page load, which would make
// the map look different each time and make our tests impossible to repeat.
// A seeded generator gives the same sequence for the same seed.

/**
 * Returns a function that produces evenly spread numbers in [0, 1).
 * The recipe is "mulberry32", a public-domain generator by Tommy Ettinger:
 * add a fixed odd constant to a 32-bit counter, then scramble its bits.
 */
export function makeRandom(seed) {
  let state = seed >>> 0;
  return function nextUniform() {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Draws one number from a bell curve with mean 0 and spread 1.
 * This is the Box-Muller method: two even random numbers become one
 * bell-curve number through a square root, a logarithm and a cosine.
 */
export function gaussian(random) {
  const first = Math.max(random(), 1e-12); // avoid log(0)
  const second = random();
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
}
