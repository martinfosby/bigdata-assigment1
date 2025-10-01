using System.Collections.Generic;

namespace task3.Core.Models;

public record PipelineParams(
    bool UsePca,
    int PcaComponents,
    string Clustering, // "kmeans"
    int K,
    string Segmentation, // "threshold","regiongrowing","watershed"
    int RegionGrowingTolerance
);

public record PipelineResult(
    string ImageName,
    string OriginalDataUrl,
    string? OverlayDataUrl,
    string? LabelsPngPath
);

public record DatasetManifest(string Name, List<string> ImagePaths);
