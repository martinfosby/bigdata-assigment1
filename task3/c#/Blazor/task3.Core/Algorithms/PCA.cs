using System;
using MathNet.Numerics.LinearAlgebra;

namespace task3.Core.Algorithms;

public class PCA
{
    public Matrix<double>? Components { get; private set; }
    public Vector<double>? Mean { get; private set; }
    public int NComponents { get; private set; }
    public Vector<double>? ExplainedVariance { get; private set; }
    public Vector<double>? ExplainedVarianceRatio { get; private set; }

    /// <summary>
    /// Fit PCA on data matrix X (n_samples x n_features).
    /// </summary>
    public void Fit(double[,] X, int nComponents)
    {
        var mX = Matrix<double>.Build.DenseOfArray(X);
        // mean vector across rows
        Mean = mX.ColumnSums() / mX.RowCount;

        // broadcast mean row to full matrix
        var meanMat = Matrix<double>.Build.Dense(mX.RowCount, mX.ColumnCount, (i, j) => Mean[j]);
        var X0 = mX - meanMat;

        var svd = X0.Svd(computeVectors: true);
        var Vt = svd.VT;
        NComponents = Math.Min(nComponents, Vt.RowCount);
        Components = Vt.SubMatrix(0, NComponents, 0, Vt.ColumnCount);

        // explained variance from singular values: (S^2) / (n-1)
        var S = svd.S;
        var vars = S.PointwisePower(2) / (mX.RowCount - 1);
        ExplainedVariance = vars.SubVector(0, NComponents);
        var totalVar = vars.Sum();
        ExplainedVarianceRatio = ExplainedVariance / totalVar;
    }

    public double[,] Transform(double[,] X)
    {
        if (Components is null || Mean is null) throw new InvalidOperationException("PCA not fitted.");
        var mX = Matrix<double>.Build.DenseOfArray(X);
        var meanMat = Matrix<double>.Build.Dense(mX.RowCount, mX.ColumnCount, (i, j) => Mean[j]);
        var X0 = mX - meanMat;
        var Z = X0 * Components.Transpose();
        return Z.ToArray();
    }

    public double[,] InverseTransform(double[,] Z)
    {
        if (Components is null || Mean is null) throw new InvalidOperationException("PCA not fitted.");
        var mZ = Matrix<double>.Build.DenseOfArray(Z);
        var Xrec = mZ * Components + Matrix<double>.Build.Dense(mZ.RowCount, Components.ColumnCount, (i, j) => Mean[j]);
        return Xrec.ToArray();
    }
}
