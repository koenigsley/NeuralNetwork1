using System;

namespace NeuralNetwork1
{
    public class Matrix
    {
        private double[,] elements;

        public int Rows => elements.GetLength(0);
        public int Columns => elements.GetLength(1);

        public double this[int row, int column]
        {
            get => elements[row, column];
            set => elements[row, column] = value;
        }

        public Matrix(int rows, int columns) : this(new double[rows, columns]) 
        {
        }

        public Matrix(double[,] elements)
        {
            this.elements = elements;
        }

        public static Matrix CreateRandom(int rows, int cols, double lowerBound, double upperBound)
        {
            var matrix = new Matrix(rows, cols);
            for (int row = 0; row < rows; row += 1)
            {
                for (int col = 0; col < cols; col += 1)
                {
                    matrix[row, col] = Randomizer.GenerateRandomDouble(lowerBound, upperBound);
                }
            }
            return matrix;
        }

        public static Matrix operator *(Matrix lhs, double[] rhs)
            => lhs * rhs.ToMatrix();

        public static Matrix operator *(Matrix lhs, Matrix rhs)
        {
            var matrix = new Matrix(lhs.Rows, rhs.Columns);
            for (int i = 0; i < lhs.Rows; i += 1)
            {
                for (int j = 0; j < rhs.Columns; j += 1)
                {
                    double sum = 0;
                    for (int r = 0; r < rhs.Rows; r += 1)
                    {
                        sum += lhs[i, r] * rhs[r, j];
                    }
                    matrix[i, j] = sum;
                }
            }
            return matrix;
        }

        public static Matrix operator +(Matrix lhs, Matrix rhs)
        {
            var matrix = new Matrix(lhs.Rows, lhs.Columns);
            for (int i = 0; i < lhs.Rows; i += 1)
            {
                for (int j = 0; j < lhs.Columns; j += 1)
                {
                    matrix[i, j] = lhs[i, j] + rhs[i, j];
                }
            }
            return matrix;
        }

        public static Matrix operator -(Matrix lhs, Matrix rhs)
        {
            var matrix = new Matrix(lhs.Rows, lhs.Columns);
            for (int i = 0; i < lhs.Rows; i += 1)
            {
                for (int j = 0; j < lhs.Columns; j += 1)
                {
                    matrix[i, j] = lhs[i, j] - rhs[i, j];
                }
            }
            return matrix;
        }

        public static Matrix operator*(Matrix lhs, double rhs)
        {
            var matrix = new Matrix(lhs.Rows, lhs.Columns);
            for (int i = 0; i < lhs.Rows; i += 1)
            {
                for (int j = 0; j < lhs.Columns; j += 1)
                {
                    matrix[i, j] = lhs[i, j] * rhs;
                }
            }
            return matrix;
        }

        public Matrix Transponse()
        {
            var matrix = new Matrix(Columns, Rows);
            for (int i = 0; i < Rows; i += 1)
            {
                for (int j = 0; j < Columns; j += 1)
                {
                    matrix[j, i] = elements[i, j];
                }
            }
            return matrix;
        }

        public static Matrix ComputeAdamarProduct(Matrix lhs, Matrix rhs)
        {
            var matrix = new Matrix(lhs.Rows, lhs.Columns);
            for (int i = 0; i < lhs.Rows; i += 1)
            {
                for (int j = 0; j < lhs.Columns; j += 1)
                {
                    matrix[i, j] = lhs[i, j] * rhs[i, j];
                }
            }
            return matrix;
        }

        public static Matrix ComputeDotProduct(Matrix lhs, double[] rhs)
        {
            var matrix = new Matrix(lhs.Rows, rhs.Length);
            for (int j = 0; j < rhs.Length; j++)
            {
                for (int k = 0; k < lhs.Rows; k++)
                {
                    matrix[k, j] = lhs[k, 0] * rhs[j];
                }
            }
            return matrix;
        }

        public double[] ToArray()
        {
            var elements = new double[Rows * Columns];
            for (int i = 0; i < Rows; i += 1)
            {
                for (int j = 0; j < Columns; j += 1)
                {
                    elements[i + j * Columns] = this.elements[i, j];
                }
            }
            return elements;
        }
    }

    public static class ArrayExtensions
    {
        public static Matrix ToMatrix(this double[] elements)
        {
            var twoDimensionalElements = new double[elements.Length, 1];
            for (int i = 0; i < elements.Length; i++)
            {
                twoDimensionalElements[i, 0] = elements[i];
            }
            return new Matrix(twoDimensionalElements);
        }
    }
}
