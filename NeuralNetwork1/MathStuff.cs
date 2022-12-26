using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    public static class MathStuff
    {
        public static double Sigmoid(double x) 
            => 1.0 / (1.0 + Math.Exp(-x));

        public static double SigmoidDerivative(double x)
        {
            double sigma = Sigmoid(x);
            return sigma * (1.0 - sigma);
        }

        public static Matrix Sigmoid(this double[] vector) 
            => vector.ToMatrix().Sigmoid();

        public static Matrix Sigmoid(this Matrix matrix)
        {
            var elements = new double[matrix.Rows, matrix.Columns];
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    elements[i, j] = Sigmoid(matrix[i, j]);
                }
            }
            return new Matrix(elements);
        }

        public static Matrix SigmoidDerivative(this double[] vector)
            => vector.ToMatrix().SigmoidDerivative();

        public static Matrix SigmoidDerivative(this Matrix matrix)
        {
            var elements = new double[matrix.Rows, matrix.Columns];
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    elements[i, j] = SigmoidDerivative(matrix[i, j]);
                }
            }
            return new Matrix(elements);
        }
    }
}
