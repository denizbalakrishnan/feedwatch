// linalg.js: the small amount of linear algebra this project needs, written by hand.
//
// A vector is a Float64Array. A matrix is an Array of Float64Array rows, so
// matrix[row][column] reads the way it is written on paper. Nothing in this
// file touches the page. It is pure arithmetic, which is what lets us test it.

/** Makes a matrix of the given size, filled with zeros. */
export function zeroMatrix(rowCount, columnCount) {
  const matrix = new Array(rowCount);
  for (let row = 0; row < rowCount; row++) matrix[row] = new Float64Array(columnCount);
  return matrix;
}

/**
 * Dot product: multiply two vectors entry by entry and add the results.
 * It is large when the vectors point the same way. The whole feed is ranked
 * by this one number, taken between "you" and each video.
 */
export function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

/** Length of a vector (Pythagoras in as many dimensions as it has). */
export function norm(vector) {
  return Math.sqrt(dot(vector, vector));
}

/** Adds weight x vector into target, in place. */
export function addScaled(target, vector, weight) {
  for (let i = 0; i < target.length; i++) target[i] += weight * vector[i];
}

/**
 * Adds weight x (vector times its own transpose) into a square matrix, in place.
 * Entry [r][c] grows by weight x vector[r] x vector[c]. Adding these up over
 * many videos builds the matrix that the ridge regression has to solve.
 */
export function addOuterProduct(matrix, vector, weight) {
  for (let row = 0; row < vector.length; row++) {
    const scaled = weight * vector[row];
    const matrixRow = matrix[row];
    for (let column = 0; column < vector.length; column++) matrixRow[column] += scaled * vector[column];
  }
}

/** Multiplies a matrix by a vector: each output entry is one row dotted with the vector. */
export function multiplyMatrixVector(matrix, vector) {
  const result = new Float64Array(matrix.length);
  for (let row = 0; row < matrix.length; row++) result[row] = dot(matrix[row], vector);
  return result;
}

/**
 * Cholesky factorisation: splits a symmetric, positive-definite matrix A into
 * L times L-transpose, where L ("lower") has zeros above the diagonal.
 * Why: a triangular system can be solved one unknown at a time, so once we
 * have L we can solve A x = b quickly and stably without inverting A.
 */
export function choleskyFactor(matrix) {
  const size = matrix.length;
  const lower = zeroMatrix(size, size);
  for (let row = 0; row < size; row++) {
    for (let column = 0; column <= row; column++) {
      let remainder = matrix[row][column];
      for (let k = 0; k < column; k++) remainder -= lower[row][k] * lower[column][k];
      if (row === column) {
        if (!(remainder > 0)) throw new Error('choleskyFactor: matrix is not positive definite');
        lower[row][row] = Math.sqrt(remainder);
      } else {
        lower[row][column] = remainder / lower[column][column];
      }
    }
  }
  return lower;
}

/** Solves L y = b from the top row down. Each row has only one new unknown. */
export function forwardSubstitute(lower, rightSide) {
  const size = lower.length;
  const solution = new Float64Array(size);
  for (let row = 0; row < size; row++) {
    let remainder = rightSide[row];
    for (let column = 0; column < row; column++) remainder -= lower[row][column] * solution[column];
    solution[row] = remainder / lower[row][row];
  }
  return solution;
}

/** Solves L-transpose x = y from the bottom row up. */
export function backSubstituteTranspose(lower, rightSide) {
  const size = lower.length;
  const solution = new Float64Array(size);
  for (let row = size - 1; row >= 0; row--) {
    let remainder = rightSide[row];
    for (let column = row + 1; column < size; column++) remainder -= lower[column][row] * solution[column];
    solution[row] = remainder / lower[row][row];
  }
  return solution;
}

/** Solves A x = b given the Cholesky factor L of A: first L y = b, then L-transpose x = y. */
export function choleskySolve(lower, rightSide) {
  return backSubstituteTranspose(lower, forwardSubstitute(lower, rightSide));
}

/**
 * Principal component analysis by power iteration.
 * Finds the directions along which a cloud of points is most spread out, so
 * that we can flatten many-dimensional video vectors onto a 2-D map while
 * keeping as much of their arrangement as possible.
 * How: multiply a vector by the covariance matrix again and again. It swings
 * towards the direction of greatest spread. For the next axis, do the same
 * while removing any part that points along the axes already found.
 * Returns the mean of the points and the unit-length axes.
 */
export function principalAxes(rows, axisCount, iterationCount = 200) {
  const dimension = rows[0].length;
  const mean = new Float64Array(dimension);
  for (const row of rows) addScaled(mean, row, 1 / rows.length);

  const covariance = zeroMatrix(dimension, dimension);
  const centred = new Float64Array(dimension);
  for (const row of rows) {
    for (let d = 0; d < dimension; d++) centred[d] = row[d] - mean[d];
    addOuterProduct(covariance, centred, 1 / rows.length);
  }

  const axes = [];
  for (let axisIndex = 0; axisIndex < axisCount; axisIndex++) {
    let axis = new Float64Array(dimension);
    for (let d = 0; d < dimension; d++) axis[d] = 1 / (d + 1 + axisIndex); // any fixed, uneven start
    for (let iteration = 0; iteration < iterationCount; iteration++) {
      axis = multiplyMatrixVector(covariance, axis);
      for (const earlier of axes) addScaled(axis, earlier, -dot(axis, earlier));
      const length = norm(axis);
      if (length < 1e-300) break; // no spread left in any new direction
      for (let d = 0; d < dimension; d++) axis[d] /= length;
    }
    axes.push(fixSign(axis));
  }
  return { mean, axes };
}

/** An axis and its opposite are equally valid. Pick the one whose biggest entry is positive. */
function fixSign(axis) {
  let biggest = 0;
  for (let d = 1; d < axis.length; d++) if (Math.abs(axis[d]) > Math.abs(axis[biggest])) biggest = d;
  if (axis[biggest] < 0) for (let d = 0; d < axis.length; d++) axis[d] = -axis[d];
  return axis;
}

/**
 * Flattens one vector onto the map axes. Each map coordinate is the vector
 * (minus an optional centre) dotted with one axis.
 */
export function projectOntoAxes(vector, axes, centre = null) {
  const coordinates = new Float64Array(axes.length);
  for (let a = 0; a < axes.length; a++) {
    coordinates[a] = dot(vector, axes[a]) - (centre ? dot(centre, axes[a]) : 0);
  }
  return coordinates;
}

/**
 * 2-D Procrustes alignment. Given the same dots in a new layout and an old
 * layout, finds the turn or mirror-flip that makes the new layout sit as
 * close as possible to the old one (least total squared distance).
 * Why: retraining can hand back the same map rotated or mirrored, because
 * the model only cares about angles between vectors. Without this the map
 * would jump every time. Returns a 2x2 matrix to apply to the new points.
 * How: in 2-D the only candidates are "rotate by some angle" or "mirror,
 * then rotate". Each has a best angle with a closed form. We keep the better.
 */
export function procrustes2D(newPoints, oldPoints) {
  // overlap[r][c] = sum over dots of old[r] x new[c]
  let m00 = 0, m01 = 0, m10 = 0, m11 = 0;
  for (let i = 0; i < newPoints.length; i++) {
    m00 += oldPoints[i][0] * newPoints[i][0];
    m01 += oldPoints[i][0] * newPoints[i][1];
    m10 += oldPoints[i][1] * newPoints[i][0];
    m11 += oldPoints[i][1] * newPoints[i][1];
  }
  const rotationFit = Math.hypot(m00 + m11, m10 - m01);
  const mirrorFit = Math.hypot(m00 - m11, m01 + m10);
  if (rotationFit >= mirrorFit) {
    const angle = Math.atan2(m10 - m01, m00 + m11);
    const c = Math.cos(angle), s = Math.sin(angle);
    return [Float64Array.of(c, -s), Float64Array.of(s, c)];
  }
  const angle = Math.atan2(m01 + m10, m00 - m11);
  const c = Math.cos(angle), s = Math.sin(angle);
  return [Float64Array.of(c, s), Float64Array.of(s, -c)];
}
