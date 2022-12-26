using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private double learningRate;
        private Layer[] layers;
        private int numLayers => layers.Length;
        private Stopwatch stopWatch;

        public StudentNetwork(int[] structure, double learningRate = 0.01)
        {
            this.learningRate = learningRate;
            layers = new Layer[structure.Length - 1];
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = Layer.CreateRandom(structure[i], structure[i + 1]);
            }

            stopWatch = new Stopwatch();
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            // Сначала надо сконструировать массивы входов и выходов
            var inputs = new double[samplesSet.Count][];
            var outputs = new double[samplesSet.Count][];

            // Теперь массивы из samplesSet группируем в inputs и outputs
            for (int i = 0; i < samplesSet.Count; ++i)
            {
                inputs[i] = samplesSet[i].input;
                outputs[i] = samplesSet[i].Output;
            }

            // Текущий счётчик эпох
            int epochToRun = 0;
            int lookedSamples = 0;
            int samplesTotal = inputs.Length * epochsCount;
            double error = double.PositiveInfinity;

            stopWatch.Restart();

            while (epochToRun < epochsCount && error > acceptableError)
            {
                epochToRun++;
                error = TrainOnDataSet(inputs, outputs, parallel);
                lookedSamples += 1;
                OnTrainProgress((double)lookedSamples / samplesTotal, error, stopWatch.Elapsed);
            }

            OnTrainProgress(1, error, stopWatch.Elapsed);
            stopWatch.Stop();

            return error;
        }

        private double TrainOnDataSet(double[][] inputs, double[][] outputs, bool parallel)
        {
            double error = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                error += Train(inputs[i], outputs[i]);
            }
            return error;
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iterations = 1;
            while (Train(sample.input, sample.Output) > acceptableError)
            {
                ++iterations;
            }
            return iterations;
        }

        private double Train(double[] input, double[] output)
        {
            var (shifts, computedOutput) = GoBackward(input, output);

            for (int i = 0; i < numLayers; i++)
            {
                layers[i].AdjustWeights(shifts[i].Item1, shifts[i].Item2, learningRate);
            }

            double error = ComputeSquaredError(computedOutput, output);
            return error;
        }

        protected override double[] Compute(double[] input)
        {
            var output = input;
            for (int i = 0; i < numLayers; i++)
            {
                output = layers[i].ComputeOutput(output);
            }
            return output;
        }

        private Tuple<Tuple<Matrix, Matrix>[], double[]> GoBackward(double[] input, double[] output)
        {
            var weightsShifts = new Matrix[numLayers];
            var biasShifts = new Matrix[numLayers];

            var activations = GoForward(input);

            var computedOutput = activations[numLayers];
            var realOutputMatrix = output.ToMatrix();
            var error = realOutputMatrix - computedOutput.ToMatrix();
            var gradient = Matrix.ComputeAdamarProduct(error, realOutputMatrix.SigmoidDerivative());

            weightsShifts[numLayers - 1] = Matrix.ComputeDotProduct(gradient, activations[activations.Length - 2]);
            biasShifts[numLayers - 1] = gradient;

            for (int i = numLayers - 1; i > 0; i--)
            {
                gradient = Matrix.ComputeAdamarProduct(
                    layers[i].Weights.Transponse() * gradient, 
                    activations[i].SigmoidDerivative()
                );
                weightsShifts[i - 1] = Matrix.ComputeDotProduct(gradient, activations[i - 1]);
                biasShifts[i - 1] = gradient;
            }

            var shifts = new Tuple<Matrix, Matrix>[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                shifts[i] = Tuple.Create(weightsShifts[i], biasShifts[i]);
            }

            return Tuple.Create(shifts, computedOutput);
        }

        private double[][] GoForward(double[] input)
        {
            var activation = input;
            var activations = new List<double[]> { activation };
            for (int i = 0; i < numLayers; i++)
            {
                activation = layers[i].ComputeOutput(activation);
                activations.Add(activation);
            }
            return activations.ToArray();
        }

        private static double ComputeSquaredError(double[] actualOutput, double[] expectedOutput)
        {
            double error = 0.0;
            for (int i = 0; i < actualOutput.Length; i++)
            {
                error += Math.Pow(actualOutput[i] - expectedOutput[i], 2.0);
            }
            error /= actualOutput.Length;
            return error;
        }
    }
}
