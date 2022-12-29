using System;

namespace NeuralNetwork1
{
    public static class Randomizer
    {
        private static readonly Random _random = new Random();

        public static double GenerateRandomDouble(double lowerBound, double upperBound) 
            => lowerBound + _random.NextDouble() * (upperBound - lowerBound);
    }
}
