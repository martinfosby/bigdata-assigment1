using System;
using FellowOakDicom;                 // FO-DICOM v5 core
using FellowOakDicom.Imaging;         // DicomPixelData
using FellowOakDicom.IO.Buffer;       // IByteBuffer

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
}
