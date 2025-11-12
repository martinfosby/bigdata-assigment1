using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using task3.Core.Algorithms;
using task3.Core.Imaging;
using task3.Core.Models;
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

            var pca = new Core.Algorithms.PCA();
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

    public PipelineResult RunPipelineMat(string imagePath, PipelineParams p)
    {
        // 1) Load image (DICOM vs standard image formats)
        var ext = Path.GetExtension(imagePath)?.ToLowerInvariant();

        Mat gray;
        if (ext == ".dcm")
        {
            gray = DicomIO.LoadGrayscaleMatSimple(imagePath);   // DICOM via fo-dicom
        }
        else
        {
            gray = ImageIO.LoadGrayscaleMat(imagePath);   // PNG/JPG/TIFF via ImageSharp
        }
        int h = gray.Rows, w = gray.Cols;


        // Assuming uploads are served from /uploads/ relative to the site root
        string originalUrl = "/uploads/" + Path.GetFileName(imagePath);


        // 3) Optional PCA (intensity-only demo to match your Python task)
        if (p.UsePca && p.PcaComponents > 0)
        {
            // 1. Convert grayscale image to float and flatten
            Mat X = new Mat();
            gray.ConvertTo(X, MatType.CV_32F);
            X = X.Reshape(1, h * w); // (N, 1)

            // 2. Compute PCA (this will just find variance along the single intensity dimension)
            Mat mean = new Mat();
            Mat eigenvectors = new Mat();
            Cv2.PCACompute(X, mean, eigenvectors, p.PcaComponents);

            // 3. Project and reconstruct
            Mat Z = new Mat();
            Cv2.PCAProject(X, mean, eigenvectors, Z);
            Mat Xrec = new Mat();
            Cv2.PCABackProject(Z, mean, eigenvectors, Xrec);

            // 4. Reshape reconstructed vector back to 2D image
            Mat imgRec = Xrec.Reshape(1, h); // back to (h, w)
            imgRec.ConvertTo(gray, MatType.CV_8U); // back to 8-bit grayscale

            // Optional: Normalize result for display
            Cv2.Normalize(gray, gray, 0, 255, NormTypes.MinMax);
        }



        
        // 4) Segmentation
        int[,] labels = new int[h, w];

        switch (p.Segmentation.ToLowerInvariant())
        {
            case "threshold":
                {
                    // Otsu threshold
                    Mat thresh = new Mat();
                    Cv2.Threshold(gray, thresh, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

                    // Convert directly to int[,] using MatIndexer
                    var idxr = thresh.GetGenericIndexer<byte>();
                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                            labels[y, x] = idxr[y, x] > 0 ? 1 : 0;
                    }
                    break;
                }

            case "regiongrowing": // Approximate with KMeans clustering
                {
                    // Flatten grayscale into Nx1 float32
                    Mat samples = gray.Reshape(1, h * w);       // flatten rows
                    samples.ConvertTo(samples, MatType.CV_32F); // ensure float

                    // Apply KMeans
                    TermCriteria criteria = new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.MaxIter, 10, 1.0);
                    Mat labelsMat = new Mat();
                    Mat centers = new Mat();
                    Cv2.Kmeans(samples, p.K, labelsMat, criteria, 3, KMeansFlags.PpCenters, centers);

                    // Reshape labelsMat into h×w
                    Mat labels2D = labelsMat.Reshape(1, h);

                    // Copy to int[,]
                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                            labels[y, x] = labels2D.At<int>(y, x);
                    }
                    break;
                }

            default: // "watershed"
                {
                    Mat srcColor = new Mat();
                    Cv2.CvtColor(gray, srcColor, ColorConversionCodes.GRAY2BGR);

                    // Threshold + distance transform
                    Mat thresh = new Mat();
                    Cv2.Threshold(gray, thresh, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

                    Mat dist = new Mat();
                    Cv2.DistanceTransform(thresh, dist, DistanceTypes.L2, DistanceTransformMasks.Mask5);
                    Cv2.Normalize(dist, dist, 0, 1.0, NormTypes.MinMax);

                    Mat sureFg = new Mat();
                    Cv2.Threshold(dist, sureFg, 0.5, 1.0, ThresholdTypes.Binary);
                    sureFg.ConvertTo(sureFg, MatType.CV_8U);

                    // Connected components for markers
                    Mat markers = new Mat();
                    Cv2.ConnectedComponents(sureFg, markers);
                    Cv2.Add(markers, Scalar.All(1), markers); // adds 1 to every element

                    // Mark unknown regions
                    Mat unknown = new Mat();
                    Cv2.Subtract(thresh, sureFg, unknown);
                    markers.SetTo(new Scalar(0), unknown); // set unknown pixels to 0

                    // Apply watershed
                    Cv2.Watershed(srcColor, markers);

                    // Copy markers to int[,]
                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                            labels[y, x] = markers.At<int>(y, x);
                    }
                    break;
                }
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
