interface DataPoint {
    features: number[];
    label?: string | number;
  }
  
  class SOM {
    private readonly weights: number[][][]; // Weights of the neurons arranged in grid form
    private readonly dimensions: number;
    private readonly rows: number;
    private readonly cols: number;
    private readonly inputDimension: number;
    private readonly learningRate: number;
    private readonly neighborhoodRadius: number;
    private readonly decayFactor: number;
  
    constructor(rows: number, cols: number, inputDimension: number, learningRate: number, neighborhoodRadius: number, decayFactor: number) {
      this.rows = rows;
      this.cols = cols;
      this.dimensions = rows * cols;
      this.inputDimension = inputDimension;
      this.learningRate = learningRate;
      this.neighborhoodRadius = neighborhoodRadius;
      this.decayFactor = decayFactor;
  
      // Initialize the weights randomly
      this.weights = Array.from({ length: rows }, (_, r) => (
        Array.from({ length: cols }, (_, c) => (
          Array.from({ length: inputDimension }, () => Math.random())
        ))
      ));
    }
  
    private calculateEuclideanDistance(a: number[], b: number[]) {
      return Math.sqrt(a.reduce((acc, val, idx) => acc + ((val - b[idx]) ** 2), 0));
    }
  
    private findBestMatchingUnit(input: number[]) {
      let minDistance = Number.POSITIVE_INFINITY;
      let bestRow = 0;
      let bestCol = 0;
  
      for (let row = 0; row < this.rows; row++) {
        for (let col = 0; col < this.cols; col++) {
          const weight = this.weights[row][col];
          const distance = this.calculateEuclideanDistance(weight, input);
          if (distance < minDistance) {
            minDistance = distance;
            bestRow = row;
            bestCol = col;
          }
        }
      }
  
      return { row: bestRow, col: bestCol };
    }
  
    private gaussianNeighborhoodFunction(row: number, col: number, bmuRow: number, bmuCol: number, radius: number) {
      const distance = this.calculateEuclideanDistance([bmuRow, bmuCol], [row, col]);
      return Math.exp(-(distance ** 2) / (2 * radius ** 2));
    }
  
    public train(data: DataPoint[], epochs: number) {
      const timeConstant = epochs / Math.log(this.neighborhoodRadius);
  
      for (let t = 0; t < epochs; t++) {
        const currentRadius = this.neighborhoodRadius * Math.exp(-t / timeConstant);
        const currentLearningRate = this.learningRate * Math.exp(-t / (this.decayFactor * epochs));
  
        // Randomly select an input from the dataset
        const input = data[Math.floor(Math.random() * data.length)].features;
  
        // Find the Best Matching Unit (BMU)
        const { row: bmuRow, col: bmuCol } = this.findBestMatchingUnit(input);
  
        // Update all neurons within the neighborhood
        for (let row = 0; row < this.rows; row++) {
          for (let col = 0; col < this.cols; col++) {
            const influence = this.gaussianNeighborhoodFunction(row, col, bmuRow, bmuCol, currentRadius);
            if (influence > 0) {
              for (let dim = 0; dim < this.inputDimension; dim++) {
                this.weights[row][col][dim] += currentLearningRate * influence * (input[dim] - this.weights[row][col][dim]);
              }
            }
          }
        }
      }
    }
  
    public getWeights(): number[][][] {
      return this.weights;
    }
  }