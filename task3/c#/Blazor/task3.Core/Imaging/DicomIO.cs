using FellowOakDicom;                 // FO-DICOM v5 core
using FellowOakDicom.Imaging;         // DicomPixelData
using FellowOakDicom.IO;
using FellowOakDicom.IO.Buffer;       // IByteBuffer
using OpenCvSharp;                 // OpenCV for .NET
using System.Drawing;

namespace task3.Core.Imaging;

public static class DicomIO
{
    /// <summary>
    /// Load a DICOM as an 8-bit grayscale byte[,] by normalizing the first frame to [0,255].
    /// Handles 8/16-bit, signed/unsigned, and MONOCHROME1 inversion.
    /// </summary>
    public static byte[,] LoadGrayscale(string path)
    {
        var dcm = DicomFile.Open(path);
        var ds  = dcm.Dataset;

        var px = DicomPixelData.Create(ds);
        int w = px.Width, h = px.Height;

        // ---- read first frame bytes into a byte[] ----
        IByteBuffer frame = px.GetFrame(0);
        int len = (int)Math.Min(int.MaxValue, frame.Size);
        var raw = new byte[len];
        frame.GetByteRange(0, len, raw);

        // Signedness: PixelRepresentation (0028,0103) == 1 => signed
        bool signed = ds.TryGetSingleValue(DicomTag.PixelRepresentation, out ushort pr) && pr == 1;

        // Bits stored
        int bitsStored = px.BitsStored;

        // Rescale slope/intercept (optional)
        double slope     = ds.TryGetSingleValue(DicomTag.RescaleSlope,     out double s)  ? s  : 1.0;
        double intercept = ds.TryGetSingleValue(DicomTag.RescaleIntercept, out double i0) ? i0 : 0.0;

        // Photometric interpretation
        string photometric = ds.TryGetSingleValue(DicomTag.PhotometricInterpretation, out string pi)
            ? pi
            : "MONOCHROME2";
        bool invert = photometric.Equals("MONOCHROME1", StringComparison.OrdinalIgnoreCase);

        var vals = new double[w * h];
        double vmin = double.PositiveInfinity, vmax = double.NegativeInfinity;

        // 8-bit frames or already 1 byte per pixel
        if (bitsStored <= 8 || raw.Length == w * h)
        {
            int count = Math.Min(raw.Length, w * h);
            for (int i = 0; i < count; i++)
            {
                double v = raw[i] * slope + intercept;
                vals[i] = v;
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
        }
        else
        {
            // Assume little-endian 16-bit samples
            int p = 0;
            for (int i = 0; i < w * h && p + 1 < raw.Length; i++)
            {
                ushort u = (ushort)(raw[p] | (raw[p + 1] << 8));
                p += 2;
                double v = signed ? (short)u : u;
                v = v * slope + intercept;
                vals[i] = v;
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
        }

        if (!double.IsFinite(vmin) || !double.IsFinite(vmax) || Math.Abs(vmax - vmin) < 1e-12)
        {
            vmin = 0; vmax = 1;
        }

        var img = new byte[h, w];
        for (int i = 0; i < w * h; i++)
        {
            double norm = (vals[i] - vmin) / (vmax - vmin);
            if (double.IsNaN(norm)) norm = 0;
            byte b = (byte)Math.Clamp(Math.Round(norm * 255.0), 0, 255);
            if (invert) b = (byte)(255 - b);
            int y = i / w, x = i % w;
            img[y, x] = b;
        }
        return img;
    }

    public static Mat LoadGrayscaleMat(string path)
    {
        // --- Step 1: Open DICOM file and extract metadata ---
        var dcm = DicomFile.Open(path);     // Load DICOM file from disk
        var ds = dcm.Dataset;               // Get the dataset (all tags)

        // --- Step 2: Access pixel data from the DICOM dataset ---
        var px = DicomPixelData.Create(ds); // Wraps pixel data
        int w = px.Width, h = px.Height;    // Image dimensions

        // Get first frame (multi-frame DICOMs can contain multiple images)
        IByteBuffer frame = px.GetFrame(0);

        // Copy frame data into a byte array
        int len = (int)Math.Min(int.MaxValue, frame.Size);
        var raw = new byte[len];
        frame.GetByteRange(0, len, raw);

        // --- Step 3: Read key image metadata ---
        bool signed = ds.TryGetSingleValue(DicomTag.PixelRepresentation, out ushort pr) && pr == 1; // 1 = signed pixels
        int bitsStored = px.BitsStored; // number of bits used per pixel (often 12–16)

        // Apply DICOM intensity calibration (if present)
        double slope = ds.TryGetSingleValue(DicomTag.RescaleSlope, out double s) ? s : 1.0;
        double intercept = ds.TryGetSingleValue(DicomTag.RescaleIntercept, out double i0) ? i0 : 0.0;

        // Handle MONOCHROME1 vs MONOCHROME2 (inverted grayscale)
        string photometric = ds.TryGetSingleValue(DicomTag.PhotometricInterpretation, out string pi)
            ? pi
            : "MONOCHROME2";
        bool invert = photometric.Equals("MONOCHROME1", StringComparison.OrdinalIgnoreCase);

        // --- Step 4: Convert raw pixel bytes into double array with scaling ---
        var vals = new double[w * h];
        double vmin = double.PositiveInfinity, vmax = double.NegativeInfinity;

        if (bitsStored <= 8 || raw.Length == w * h)
        {
            // --- Case 1: 8-bit pixels ---
            int count = Math.Min(raw.Length, w * h);
            for (int i = 0; i < count; i++)
            {
                // Apply rescale slope + intercept
                double v = raw[i] * slope + intercept;
                vals[i] = v;

                // Track min/max for normalization
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
        }
        else
        {
            // --- Case 2: 16-bit pixels ---
            int p = 0;
            for (int i = 0; i < w * h && p + 1 < raw.Length; i++)
            {
                // Combine two bytes into one 16-bit value (little-endian)
                ushort u = (ushort)(raw[p] | (raw[p + 1] << 8));
                p += 2;

                // Interpret as signed or unsigned
                double v = signed ? (short)u : u;

                // Apply DICOM scaling
                v = v * slope + intercept;
                vals[i] = v;

                // Track intensity range
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
        }

        // --- Step 5: Validate min/max and avoid division by zero ---
        if (!double.IsFinite(vmin) || !double.IsFinite(vmax) || Math.Abs(vmax - vmin) < 1e-12)
        {
            vmin = 0;
            vmax = 1;
        }

        // --- Step 6: Create an 8-bit grayscale Mat and normalize pixel values ---
        Mat mat = new Mat(h, w, MatType.CV_8U);
        for (int i = 0; i < w * h; i++)
        {
            // Normalize to [0, 1]
            double norm = (vals[i] - vmin) / (vmax - vmin);
            if (double.IsNaN(norm)) norm = 0;

            // Scale to 8-bit grayscale range [0, 255]
            byte b = (byte)Math.Clamp(Math.Round(norm * 255.0), 0, 255);

            // Apply inversion if needed (MONOCHROME1)
            if (invert)
                b = (byte)(255 - b);

            // Compute pixel coordinates and write to Mat
            int y = i / w;
            int x = i % w;
            mat.Set(y, x, b);
        }

        // --- Step 7: Return the OpenCV Mat ---
        return mat;
    }

    /// <summary>
    /// Loads a DICOM as a grayscale OpenCV Mat using Pixels array directly.
    /// Cross-platform, avoids Bitmap entirely.
    /// </summary>

    public static Mat LoadGrayscaleMatSimple(string path)
    {
        // Step 1: Load DICOM and render it
        var dicomImage = new DicomImage(path);
        var rendered = dicomImage.RenderImage();
        var pixelsArray = rendered.Pixels;

        // Step 2: Determine image dimensions
        int width = rendered.Width;
        int height = rendered.Height;

        // Step 3: Allocate a byte array for pixel data
        byte[] pixels = new byte[width * height]; // your pixel data
        Mat mat = new Mat(height, width, MatType.CV_8UC1);

        // Step 4: Copy pixel data into the byte array
        unsafe
        {
            byte* ptr = (byte*)pixelsArray.Pointer;
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = ptr[i];
            }
        }

        // Copy pixel values safely using Set or SetArray
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                mat.Set(y, x, pixels[y * width + x]);
            }
        }

        return mat;
    }



}
