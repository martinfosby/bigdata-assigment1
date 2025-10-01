using System;
using System.Collections.Generic;

namespace task3.Core.Algorithms;

public static class Watershed
{
    // Simple city-block distance transform for binary mask (1=foreground, 0=background)
    public static int[,] DistanceTransform(int[,] mask)
    {
        int h = mask.GetLength(0), w = mask.GetLength(1);
        var d = new int[h, w];
        int INF = 1 << 29;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                d[y, x] = mask[y, x] > 0 ? INF : 0;

        // forward
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (d[y, x] == 0) continue;
                int v = d[y, x];
                if (x > 0) v = Math.Min(v, d[y, x - 1] + 1);
                if (y > 0) v = Math.Min(v, d[y - 1, x] + 1);
                d[y, x] = v;
            }
        }
        // backward
        for (int y = h - 1; y >= 0; y--)
        {
            for (int x = w - 1; x >= 0; x--)
            {
                if (d[y, x] == 0) continue;
                int v = d[y, x];
                if (x + 1 < w) v = Math.Min(v, d[y, x + 1] + 1);
                if (y + 1 < h) v = Math.Min(v, d[y + 1, x] + 1);
                d[y, x] = v;
            }
        }
        return d;
    }

    // Find local maxima as initial markers
    public static int[,] FindMarkers(int[,] dist, int minDistance = 2)
    {
        int h = dist.GetLength(0), w = dist.GetLength(1);
        var markers = new int[h, w];
        int label = 1;
        for (int y = 1; y < h-1; y++)
        {
            for (int x = 1; x < w-1; x++)
            {
                int v = dist[y, x];
                if (v <= 0) continue;
                bool isMax = true;
                for (int yy = y-1; yy <= y+1 && isMax; yy++)
                for (int xx = x-1; xx <= x+1 && isMax; xx++)
                {
                    if (yy==y && xx==x) continue;
                    if (dist[yy, xx] > v) isMax = false;
                }
                if (!isMax) continue;

                // suppress too-close peaks
                bool near = false;
                for (int yy = Math.Max(0, y - minDistance); yy <= Math.Min(h - 1, y + minDistance) && !near; yy++)
                for (int xx = Math.Max(0, x - minDistance); xx <= Math.Min(w - 1, x + minDistance) && !near; xx++)
                    if (markers[yy, xx] > 0) near = true;
                if (near) continue;

                markers[y, x] = label++;
            }
        }
        return markers;
    }

    // Priority flood from markers on distance map to split touching objects
    public static int[,] Segment(byte[,] image, int minDistance = 2)
    {
        int h = image.GetLength(0), w = image.GetLength(1);
        // Binary mask via Otsu
        var bin = OtsuThreshold.Segment(image);
        // Distance on foreground (invert: 1 where bin==1)
        var mask = new int[h, w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                mask[y, x] = bin[y, x] > 0 ? 1 : 0;

        var dist = DistanceTransform(mask);
        var markers = FindMarkers(dist, minDistance);
        var labels = new int[h, w];
        Array.Copy(markers, labels, markers.Length);

        var pq = new PriorityQueue<(int x,int y), int>(); // max-heap via negative priority
        var inQueue = new bool[h, w];
        var dirs = new (int dx,int dy)[]{(1,0),(-1,0),(0,1),(0,-1)};

        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                if (markers[y, x] > 0)
                {
                    pq.Enqueue((x, y), -dist[y, x]);
                    inQueue[y, x] = true;
                }

        while (pq.Count > 0)
        {
            var (x, y) = pq.Dequeue();
            int lab = labels[y, x];
            foreach (var (dx,dy) in dirs)
            {
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                if (mask[ny, nx] == 0) continue; // stay in foreground
                if (labels[ny, nx] == 0)
                {
                    labels[ny, nx] = lab;
                    if (!inQueue[ny, nx])
                    {
                        pq.Enqueue((nx, ny), -dist[ny, nx]);
                        inQueue[ny, nx] = true;
                    }
                }
                else if (labels[ny, nx] != lab)
                {
                    // boundary: set to 0 (watershed line)
                    labels[y, x] = 0;
                }
            }
        }
        return labels;
    }
}
