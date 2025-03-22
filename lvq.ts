interface LabeledDataPoint extends DataPoint {
    label: string | number;
  }
  
  class LVQ {
    private readonly prototypes: number[][]; // Prototypes for each class
    private readonly classes: Set<string | number>; // Unique labels for classification
    private readonly learningRate: number;
    private readonly decayFactor: number;
  
    constructor(classes: (string | number)[], learningRate: number, decayFactor: number) {
      this.classes = new Set(classes);
      this.learningRate = learningRate;
      this.decayFactor = decayFactor;
  
      // Initialize prototypes randomly
      this.prototypes = Array.from({ length: classes.length }, () => (
        Array.from({ length: 2 }, () => Math.random()) // Assuming 2-dimensional input
      ));
    }
  
    private calculateEuclideanDistance(a: number[], b: number[]) {
      return Math.sqrt(a.reduce((acc, val, idx) => acc + ((val - b[idx]) ** 2), 0));
    }
  
    private findNearestPrototype(input: number[]) {
      let minDistance = Number.POSITIVE_INFINITY;
      let nearestPrototypeIndex = 0;
  
      for (let i = 0; i < this.prototypes.length; i++) {
        const distance = this.calculateEuclideanDistance(input, this.prototypes[i]);
        if (distance < minDistance) {
          minDistance = distance;
          nearestPrototypeIndex = i;
        }
      }
  
      return nearestPrototypeIndex;
    }
  
    public train(data: LabeledDataPoint[], epochs: number) {
      for (let t = 0; t < epochs; t++) {
        const currentLearningRate = this.learningRate * Math.exp(-t / (this.decayFactor * epochs));
  
        // Randomly select an input from the dataset
        const example = data[Math.floor(Math.random() * data.length)];
        const input = example.features;
        const label = example.label;
  
        // Find the nearest prototype
        const nearestPrototypeIndex = this.findNearestPrototype(input);
  
        // Adjust the prototype based on whether it belongs to the same class or not
        if (label === this.classes[nearestPrototypeIndex]) {
          // Move prototype closer to the input
          for (let dim = 0; dim < input.length; dim++) {
            this.prototypes[nearestPrototypeIndex][dim] += currentLearningRate * (input[dim] - this.prototypes[nearestPrototypeIndex][dim]);
          }
        } else {
          // Move prototype further away from the input
          for (let dim = 0; dim < input.length; dim++) {
            this.prototypes[nearestPrototypeIndex][dim] -= currentLearningRate * (input[dim] - this.prototypes[nearestPrototypeIndex][dim]);
          }
        }
      }
    }
  
    public classify(input: number[]) {
      const nearestPrototypeIndex = this.findNearestPrototype(input);
      return this.classes[nearestPrototypeIndex];
    }
  
    public getPrototypes(): number[][] {
      return this.prototypes;
    }
  }