using System;

namespace task3.Core.Algorithms;

public static class OtsuThreshold
{
    public static int Compute(byte[,] image)
    {
        int h = image.GetLength(0), w = image.GetLength(1);
        var hist = new int[256];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                hist[image[y, x]]++;

        int total = h * w;
        double sum = 0;
        for (int t = 0; t < 256; t++) sum += t * hist[t];
        double sumB = 0;
        int wB = 0;
        int wF = 0;
        double maxVar = 0;
        int threshold = 0;

        for (int t = 0; t < 256; t++)
        {
            wB += hist[t];
            if (wB == 0) continue;
            wF = total - wB;
            if (wF == 0) break;

            sumB += t * hist[t];
            double mB = sumB / wB;
            double mF = (sum - sumB) / wF;
            double between = wB * wF * (mB - mF) * (mB - mF);
            if (between > maxVar)
            {
                maxVar = between;
                threshold = t;
            }
        }
        return threshold;
    }

    public static int[,] Segment(byte[,] image)
    {
        int h = image.GetLength(0), w = image.GetLength(1);
        int T = Compute(image);
        var labels = new int[h, w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                labels[y, x] = image[y, x] >= T ? 1 : 0;
        return labels;
    }
}
