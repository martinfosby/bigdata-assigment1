using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace task3.Core.Imaging;

public static class ImageIO
{
    public static byte[,] LoadGrayscale(string path)
    {
        using var img = Image.Load(path);
        img.Mutate(x => x.Grayscale());
        var g = img.CloneAs<L8>();
        int h = g.Height, w = g.Width;
        var data = new byte[h, w];
        g.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                    data[y, x] = row[x].PackedValue;
            }
        });
        return data;
    }

    public static void SaveGrayscale(byte[,] data, string path)
    {
        int h = data.GetLength(0), w = data.GetLength(1);
        using var img = new Image<L8>(w, h);
        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                    row[x].PackedValue = data[y, x];
            }
        });
        img.Save(path);
    }

    public static void SaveLabelMap(int[,] labels, string path, int alpha = 200)
    {
        // Random but stable palette, 0 reserved as black/border
        int h = labels.GetLength(0), w = labels.GetLength(1);
        using var img = new Image<Rgba32>(w, h);
        var rand = new Random(1234);
        var palette = new Rgba32[256];
        palette[0] = new Rgba32(0, 0, 0, 255);
        for (int i = 1; i < palette.Length; i++)
            palette[i] = new Rgba32((byte)rand.Next(30,255), (byte)rand.Next(30,255), (byte)rand.Next(30,255), (byte)alpha);

        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    int lab = labels[y, x];
                    if (lab < 0) lab = 0;
                    if (lab > 255) lab = 255;
                    row[x] = palette[lab];
                }
            }
        });
        img.Save(path);
    }

    public static string ToDataUrl(byte[,] data)
    {
        using var ms = new MemoryStream();
        int h = data.GetLength(0), w = data.GetLength(1);
        using var img = new Image<L8>(w, h);
        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                    row[x].PackedValue = data[y, x];
            }
        });
        img.SaveAsPng(ms);
        var b64 = Convert.ToBase64String(ms.ToArray());
        return $"data:image/png;base64,{b64}";
    }

    public static string LabelMapToDataUrl(int[,] labels, int alpha = 180)
    {
        using var ms = new MemoryStream();
        int h = labels.GetLength(0), w = labels.GetLength(1);
        using var img = new Image<Rgba32>(w, h);
        var rand = new Random(1234);
        var palette = new Rgba32[256];
        palette[0] = new Rgba32(0, 0, 0, 0);
        for (int i = 1; i < palette.Length; i++)
            palette[i] = new Rgba32((byte)rand.Next(30,255), (byte)rand.Next(30,255), (byte)rand.Next(30,255), (byte)alpha);

        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    int lab = labels[y, x];
                    if (lab < 0) lab = 0;
                    if (lab > 255) lab = 255;
                    row[x] = palette[lab];
                }
            }
        });
        img.SaveAsPng(ms);
        var b64 = Convert.ToBase64String(ms.ToArray());
        return $"data:image/png;base64,{b64}";
    }
}
