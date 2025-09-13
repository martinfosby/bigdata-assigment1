using System;

namespace task3.Core.Algorithms;

public class KMeans
{
    private readonly int _k;
    private readonly int _maxIter;
    private readonly int _seed;
    public double[][]? Centers { get; private set; }
    public int[]? Labels { get; private set; }

    public KMeans(int k, int maxIter = 100, int seed = 42)
    {
        _k = k;
        _maxIter = maxIter;
        _seed = seed;
    }

    public void Fit(double[][] X)
    {
        int n = X.Length, d = X[0].Length;
        var rnd = new Random(_seed);
        Centers = new double[_k][];
        var chosen = new System.Collections.Generic.HashSet<int>();
        for (int i = 0; i < _k; i++)
        {
            int idx;
            do { idx = rnd.Next(n); } while (!chosen.Add(idx));
            Centers[i] = (double[])X[idx].Clone();
        }

        Labels = new int[n];
        bool changed = true;
        int iter = 0;
        var counts = new int[_k];
        var sums = new double[_k][];
        for (int i = 0; i < _k; i++) sums[i] = new double[d];

        while (changed && iter++ < _maxIter)
        {
            changed = false;
            // assign
            for (int i = 0; i < n; i++)
            {
                int best = 0;
                double bestDist = double.PositiveInfinity;
                for (int c = 0; c < _k; c++)
                {
                    double dist = 0;
                    var center = Centers![c];
                    var xi = X[i];
                    for (int j = 0; j < d; j++)
                    {
                        double diff = xi[j] - center[j];
                        dist += diff * diff;
                    }
                    if (dist < bestDist) { bestDist = dist; best = c; }
                }
                if (Labels[i] != best) { Labels[i] = best; changed = true; }
            }

            // recompute centers
            Array.Fill(counts, 0);
            for (int c = 0; c < _k; c++) Array.Fill(sums[c], 0.0);
            for (int i = 0; i < n; i++)
            {
                int lab = Labels[i];
                counts[lab]++;
                var xi = X[i];
                var sumc = sums[lab];
                for (int j = 0; j < d; j++) sumc[j] += xi[j];
            }
            for (int c = 0; c < _k; c++)
            {
                if (counts[c] == 0) continue;
                for (int j = 0; j < d; j++)
                    Centers![c][j] = sums[c][j] / counts[c];
            }
        }
    }
}
