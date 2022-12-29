using System;

namespace NeuralNetwork1
{
    public class Layer
    {
        private Matrix weights, bias;

        public Matrix Weights => weights;

        public Layer(Matrix weights, Matrix bias)
        {
            this.weights = weights;
            this.bias = bias;
        }

        public static Layer CreateRandom(int inputSize, int outputSize, double weightsLowerBound = -1.0, double weightsUpperBound = 1.0, double biasLowerBound = -1.0, double biasUpperBound = 1.0)
        {
            var weights = Matrix.CreateRandom(outputSize, inputSize, weightsLowerBound, weightsUpperBound);

            var biasElements = new double[outputSize, 1];
            var b = Randomizer.GenerateRandomDouble(biasLowerBound, biasUpperBound);
            for (int i = 0; i < outputSize; i++)
            {
                biasElements[i, 0] = b;
            }
            var bias = new Matrix(biasElements);

            return new Layer(weights, bias);
        }

        public void AdjustWeights(Matrix weightsShift, Matrix biasShift, double learningRate)
        {
            weights = weights + (weightsShift * learningRate);
            bias = bias + (biasShift * learningRate);
        }

        public double[] ComputeOutput(double[] input) 
            => ((weights * input) + bias).Sigmoid().ToArray();
    }
}
