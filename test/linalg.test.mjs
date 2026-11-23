// Tests for linalg.js. Run with: node --test
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  zeroMatrix, dot, norm, choleskyFactor, choleskySolve, principalAxes, projectOntoAxes, procrustes2D,
} from '../js/linalg.js';
import { makeRandom, gaussian } from '../js/random.js';

const close = (actual, expected, tolerance = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} is not within ${tolerance} of ${expected}`);

const toMatrix = (rows) => rows.map((row) => Float64Array.from(row));

test('choleskyFactor reproduces a textbook factorisation', () => {
  // A classic example: A = L L-transpose with L = [[2,0,0],[6,1,0],[-8,5,3]].
  const lower = choleskyFactor(toMatrix([[4, 12, -16], [12, 37, -43], [-16, -43, 98]]));
  const expected = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]];
  for (let row = 0; row < 3; row++) for (let column = 0; column < 3; column++) close(lower[row][column], expected[row][column]);
});

test('choleskySolve finds the x we started from', () => {
  const matrix = toMatrix([[4, 12, -16], [12, 37, -43], [-16, -43, 98]]);
  // With x = [1, 2, 3], A x = [-20, -43, 192].
  const solution = choleskySolve(choleskyFactor(matrix), Float64Array.of(-20, -43, 192));
  close(solution[0], 1);
  close(solution[1], 2);
  close(solution[2], 3);
});

test('choleskyFactor refuses a matrix that is not positive definite', () => {
  assert.throws(() => choleskyFactor(toMatrix([[1, 2], [2, 1]])), /not positive definite/);
});

test('principalAxes finds the direction a cloud is stretched along', () => {
  const random = makeRandom(7);
  const rows = [];
  for (let i = 0; i < 400; i++) {
    const along = 5 * gaussian(random); // big spread along (0.6, 0.8)
    const across = 0.3 * gaussian(random); // small spread along (-0.8, 0.6)
    rows.push(Float64Array.of(10 + 0.6 * along - 0.8 * across, -4 + 0.8 * along + 0.6 * across));
  }
  const { mean, axes } = principalAxes(rows, 2);
  close(mean[0], 10, 0.6);
  close(mean[1], -4, 0.6);
  close(Math.abs(dot(axes[0], Float64Array.of(0.6, 0.8))), 1, 1e-3);
  close(norm(axes[0]), 1);
  close(norm(axes[1]), 1);
  close(dot(axes[0], axes[1]), 0);
});

test('projectOntoAxes measures from the centre it is given', () => {
  const axes = [Float64Array.of(1, 0, 0), Float64Array.of(0, 1, 0)];
  const flat = projectOntoAxes(Float64Array.of(3, 4, 5), axes, Float64Array.of(1, 1, 1));
  close(flat[0], 2);
  close(flat[1], 3);
});

function turned(points, matrix) {
  return points.map((point) => Float64Array.of(
    matrix[0][0] * point[0] + matrix[0][1] * point[1],
    matrix[1][0] * point[0] + matrix[1][1] * point[1],
  ));
}

test('procrustes2D undoes a rotation', () => {
  const random = makeRandom(3);
  const oldPoints = Array.from({ length: 50 }, () => Float64Array.of(gaussian(random), gaussian(random)));
  const angle = 1.1;
  const newPoints = turned(oldPoints, [[Math.cos(angle), -Math.sin(angle)], [Math.sin(angle), Math.cos(angle)]]);
  const aligned = turned(newPoints, procrustes2D(newPoints, oldPoints));
  for (let i = 0; i < oldPoints.length; i++) {
    close(aligned[i][0], oldPoints[i][0]);
    close(aligned[i][1], oldPoints[i][1]);
  }
});

test('procrustes2D undoes a mirror flip', () => {
  const random = makeRandom(4);
  const oldPoints = Array.from({ length: 50 }, () => Float64Array.of(gaussian(random), gaussian(random)));
  const newPoints = turned(oldPoints, [[0.6, 0.8], [0.8, -0.6]]); // a reflection
  const aligned = turned(newPoints, procrustes2D(newPoints, oldPoints));
  for (let i = 0; i < oldPoints.length; i++) {
    close(aligned[i][0], oldPoints[i][0]);
    close(aligned[i][1], oldPoints[i][1]);
  }
});

test('zeroMatrix rows are independent', () => {
  const matrix = zeroMatrix(2, 2);
  matrix[0][0] = 5;
  assert.equal(matrix[1][0], 0);
});
