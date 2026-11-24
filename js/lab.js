// lab.js: draws the training-error chart in the lab.
// Presentation only: it is handed the numbers and draws two lines.

const MARGIN = { left: 46, right: 132, top: 12, bottom: 26 };

/**
 * Draws how wrong the model was after each training sweep, on the cells it
 * trained on and on the cells that were hidden from it.
 * history is a list of { sweep, trainingError, heldOutError }.
 */
export function drawErrorChart(canvas, history, sweepCount, theme) {
  const box = canvas.getBoundingClientRect();
  if (!box.width) return; // the panel is closed, so there is nothing to draw on
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = Math.round(box.width * pixelRatio);
  canvas.height = Math.round(box.height * pixelRatio);
  const context = canvas.getContext('2d');
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, box.width, box.height);

  const plotWidth = box.width - MARGIN.left - MARGIN.right;
  const plotHeight = box.height - MARGIN.top - MARGIN.bottom;
  let highest = 0.05;
  for (const point of history) highest = Math.max(highest, point.trainingError, point.heldOutError);
  const xOf = (sweep) => MARGIN.left + (plotWidth * (sweep - 1)) / Math.max(1, sweepCount - 1);
  const yOf = (error) => MARGIN.top + plotHeight * (1 - error / highest);

  // the two axes and their end labels
  context.strokeStyle = theme.dot;
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(MARGIN.left, MARGIN.top);
  context.lineTo(MARGIN.left, MARGIN.top + plotHeight);
  context.lineTo(MARGIN.left + plotWidth, MARGIN.top + plotHeight);
  context.stroke();
  context.fillStyle = theme.dot;
  context.font = '12px system-ui, sans-serif';
  context.textBaseline = 'middle';
  context.textAlign = 'right';
  context.fillText(highest.toFixed(2), MARGIN.left - 6, MARGIN.top);
  context.fillText('0', MARGIN.left - 6, MARGIN.top + plotHeight);
  context.textBaseline = 'top';
  context.textAlign = 'left';
  context.fillText('sweep 1', MARGIN.left, MARGIN.top + plotHeight + 6);
  context.textAlign = 'right';
  context.fillText(`sweep ${sweepCount}`, MARGIN.left + plotWidth, MARGIN.top + plotHeight + 6);

  if (history.length === 0) return;
  const last = history[history.length - 1];
  let trainedLabelY = yOf(last.trainingError);
  let hiddenLabelY = yOf(last.heldOutError);
  if (Math.abs(trainedLabelY - hiddenLabelY) < 14) { // keep the two labels from sitting on top of each other
    const middle = (trainedLabelY + hiddenLabelY) / 2;
    hiddenLabelY = middle + (last.heldOutError >= last.trainingError ? -8 : 8);
    trainedLabelY = middle + (last.heldOutError >= last.trainingError ? 8 : -8);
  }

  drawLine(context, history.map((point) => [xOf(point.sweep), yOf(point.trainingError)]), theme.text, [5, 4]);
  drawLine(context, history.map((point) => [xOf(point.sweep), yOf(point.heldOutError)]), theme.you, []);
  context.textAlign = 'left';
  context.textBaseline = 'middle';
  context.fillStyle = theme.text;
  context.fillText('cells it trained on', xOf(last.sweep) + 8, trainedLabelY);
  context.fillStyle = theme.you;
  context.fillText('cells hidden from it', xOf(last.sweep) + 8, hiddenLabelY);
}

function drawLine(context, points, colour, dash) {
  context.strokeStyle = colour;
  context.lineWidth = 2;
  context.setLineDash(dash);
  context.beginPath();
  points.forEach(([x, y], index) => (index === 0 ? context.moveTo(x, y) : context.lineTo(x, y)));
  context.stroke();
  context.setLineDash([]);
  context.fillStyle = colour;
  for (const [x, y] of points) {
    context.beginPath();
    context.arc(x, y, 2.5, 0, 2 * Math.PI);
    context.fill();
  }
}
