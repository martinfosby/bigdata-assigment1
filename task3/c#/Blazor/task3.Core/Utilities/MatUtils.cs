using OpenCvSharp;

namespace task3.Core.Utilities;
public static class MatUtils
{
    public static unsafe Mat ByteArrayToMat(byte[,] data)
    {
        int h = data.GetLength(0);
        int w = data.GetLength(1);

        // Create a single-channel 8-bit Mat
        Mat mat = new Mat(h, w, MatType.CV_8UC1);

        // Pointer to the Mat data
        byte* ptr = mat.DataPointer;

        // Copy each pixel (linearize 2D indices)
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                ptr[y * w + x] = data[y, x];
            }
        }

        return mat;
    }
}
