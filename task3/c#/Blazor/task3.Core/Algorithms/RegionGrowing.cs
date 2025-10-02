using System;
using System.Collections.Generic;

namespace task3.Core.Algorithms;

public static class RegionGrowing
{
    public static int[,] Grow(byte[,] image, List<(int x,int y)> seeds, int tolerance = 15, int maxLabelCount = 250)
    {
        int h = image.GetLength(0), w = image.GetLength(1);
        var labels = new int[h, w];
        int label = 1;
        var q = new Queue<(int x,int y)>();
        var dirs = new (int dx,int dy)[]{(1,0),(-1,0),(0,1),(0,-1)};

        foreach (var seed in seeds)
        {
            if (label > maxLabelCount) break;
            int sx = seed.x, sy = seed.y;
            if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
            if (labels[sy, sx] != 0) continue;
            int mean = image[sy, sx];
            int count = 1;
            labels[sy, sx] = label;
            q.Enqueue((sx,sy));

            while (q.Count > 0)
            {
                var (x,y) = q.Dequeue();
                foreach (var (dx,dy) in dirs)
                {
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    if (labels[ny, nx] != 0) continue;
                    int val = image[ny, nx];
                    int diff = Math.Abs(val - mean);
                    if (diff <= tolerance)
                    {
                        labels[ny, nx] = label;
                        q.Enqueue((nx, ny));
                        // online update of region mean
                        mean = (mean * count + val) / (++count);
                    }
                }
            }
            label++;
        }
        return labels;
    }
}
