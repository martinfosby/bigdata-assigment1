using System;

namespace task3.Core.Imaging;

public static class ArrayUtils
{
    public static double[,] ToDouble2D(byte[,] input)
    {
        int h = input.GetLength(0);
        int w = input.GetLength(1);
        var result = new double[h, w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                result[y, x] = input[y, x];
        return result;
    }

    public static byte[,] ToByte2D(double[,] input, double min = 0, double max = 255)
    {
        int h = input.GetLength(0);
        int w = input.GetLength(1);
        var result = new byte[h, w];
        double range = max - min;
        if (range <= 0) range = 1;

        // auto scale to [0,255] if min>=max sentinel passed
        double vmin = double.PositiveInfinity, vmax = double.NegativeInfinity;
        if (min >= max)
        {
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double v = input[y, x];
                    if (v < vmin) vmin = v;
                    if (v > vmax) vmax = v;
                }
            min = vmin;
            max = vmax;
            range = max - min;
            if (range <= 1e-12) range = 1;
        }

        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double v = (input[y, x] - min) / range;
                v = Math.Clamp(v, 0, 1);
                result[y, x] = (byte)Math.Round(v * 255.0);
            }
        return result;
    }

    public static T[,] Create2D<T>(int h, int w, T value = default!)
    {
        var arr = new T[h, w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                arr[y, x] = value;
        return arr;
    }

    public static int[,] Create2DInt(int h, int w, int value = 0) => Create2D<int>(h, w, value);

    public static (int H, int W) Shape<T>(T[,] a) => (a.GetLength(0), a.GetLength(1));
}
