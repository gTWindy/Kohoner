// Пример1

const rows = 8;
const cols = 8;
const inputDimension = 2;
const learningRate = 0.1;
const neighborhoodRadius = 6;
let decayFactor: number = 0.05;
// Количество эпох
let epochs: number = 3000;

const som = new SOM(rows, cols, inputDimension, learningRate, neighborhoodRadius, decayFactor);

// Генерируемый случайный набор данных
const dataPoints: DataPoint[] = [
  { features: [-1, -1] },
  { features: [1, 1] },
  { features: [0, 0] },
  { features: [-0.5, -0.5] },
  { features: [0.5, 0.5] },
  { features: [0.25, 0.25] },
  { features: [-0.25, -0.25] },
  { features: [0.75, 0.75] },
  { features: [-0.75, -0.75] },
  { features: [0.125, 0.125] },
  { features: [-0.125, -0.125] },
];

som.train(dataPoints, epochs);

console.log(`Тренерованные веса:\n${JSON.stringify(som.getWeights(), null, 2)}`);

// Пример2
const classes = ["A", "B"];
decayFactor = 0.02;
epochs = 500;

const lvq = new LVQ(classes, learningRate, decayFactor);

// Генерируемый случайный набор данных с метками
const labeledDataPoints: LabeledDataPoint[] = [
  { features: [-1, -1], label: "A" },
  { features: [1, 1], label: "B" },
  { features: [0, 0], label: "A" },
  { features: [-0.5, -0.5], label: "A" },
  { features: [0.5, 0.5], label: "B" },
  { features: [0.25, 0.25], label: "B" },
  { features: [-0.25, -0.25], label: "A" },
  { features: [0.75, 0.75], label: "B" },
  { features: [-0.75, -0.75], label: "A" },
  { features: [0.125, 0.125], label: "B" },
  { features: [-0.125, -0.125], label: "A" },
];

lvq.train(labeledDataPoints, epochs);

console.log(`Тренерованные веса:\n${JSON.stringify(lvq.getPrototypes(), null, 2)}`);