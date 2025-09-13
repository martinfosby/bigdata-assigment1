using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using task3.Core.Algorithms;
using task3.Core.Imaging;
using task3.Core.Models;

namespace task3.BlazorServer;

public class PipelineService
{
    private readonly string _dataDir;

    public PipelineService()
    {
        _dataDir = Path.Combine(AppContext.BaseDirectory, "AppData");
        Directory.CreateDirectory(_dataDir);
    }

    /// <summary>
    /// Save uploads to disk unchanged (supports .dcm, .png, .jpg, etc.) using async copy.
    /// </summary>
    public async Task<IEnumerable<string>> SaveUploadsAsync(IEnumerable<(string name, Stream stream)> uploads)
    {
        var saved = new List<string>();

        foreach (var (name, stream) in uploads)
        {
            var outPath = Path.Combine(_dataDir, Path.GetFileName(name)); // keep original name & extension
            await using var fs = File.Create(outPath);
            await stream.CopyToAsync(fs);
            saved.Add(outPath);
        }

        return saved;
    }

    public PipelineResult RunPipeline(string imagePath, PipelineParams p)
    {
        // 1) Load image (DICOM vs standard image formats)
        byte[,] gray;
        var ext = Path.GetExtension(imagePath)?.ToLowerInvariant();

        if (ext == ".dcm")
        {
            gray = DicomIO.LoadGrayscale(imagePath);   // DICOM via fo-dicom
        }
        else
        {
            gray = ImageIO.LoadGrayscale(imagePath);   // PNG/JPG/TIFF via ImageSharp
        }

        // 2) Original preview (data URL)
        var originalUrl = ImageIO.ToDataUrl(gray);
        int h = gray.GetLength(0), w = gray.GetLength(1);

        // 3) Optional PCA (intensity-only demo to match your Python task)
        if (p.UsePca && p.PcaComponents > 0)
        {
            var X = new double[h * w, 1];
            int idx = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    X[idx++, 0] = gray[y, x];

            var pca = new PCA();
            pca.Fit(X, Math.Min(p.PcaComponents, 1));
            var Z = pca.Transform(X);
            var Xrec = pca.InverseTransform(Z);

            idx = 0;
            var rec = new byte[h, w];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    rec[y, x] = (byte)Math.Clamp(Math.Round(Xrec[idx++, 0]), 0, 255);

            gray = rec;
        }

        // 4) Segmentation
        int[,] labels;
        switch (p.Segmentation.ToLowerInvariant())
        {
            case "threshold":
                labels = OtsuThreshold.Segment(gray);
                break;

            case "regiongrowing":
            {
                // Simple auto seeds: pick K brightest pixels
                var flat = new double[h * w][];
                int i = 0;
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        flat[i++] = new double[] { gray[y, x] };

                var km = new KMeans(p.K);
                km.Fit(flat);

                var seeds = Enumerable.Range(0, h * w)
                    .OrderByDescending(id => flat[id][0])
                    .Take(p.K)
                    .Select(id => (x: id % w, y: id / w))
                    .ToList();

                labels = RegionGrowing.Grow(gray, seeds, p.RegionGrowingTolerance);
                break;
            }

            default: // "watershed"
                labels = Watershed.Segment(gray, minDistance: 2);
                break;
        }

        // 5) Overlay & file export
        var overlay = ImageIO.LabelMapToDataUrl(labels, 160);
        var outPng = Path.Combine(
            Path.GetDirectoryName(imagePath)!,
            Path.GetFileNameWithoutExtension(imagePath) + "_labels.png");
        ImageIO.SaveLabelMap(labels, outPng, 200);

        return new PipelineResult(Path.GetFileName(imagePath), originalUrl, overlay, outPng);
    }
}
