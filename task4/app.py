import os
import re
import zipfile
import tempfile
import io
import random
import numpy as np
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

import pydicom
from skimage import io as ski_io, color, exposure, transform, filters, measure, segmentation
from skimage.restoration import denoise_bilateral
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.preprocessing import StandardScaler
from skimage.measure import marching_cubes
import trimesh
from skimage.transform import resize
from scipy.ndimage import binary_dilation, binary_closing, label as cc_label

try:
    import SimpleITK as sitk
    HAS_SITK = True
except Exception:
    HAS_SITK = False

# ---- Config ----
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "dcm", "zip"}
MAX_CONTENT = 500 * 1024 * 1024  # 500MB

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT

random.seed(0)
np.random.seed(0)

# ---- Utils ----
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# ---- Loaders ----
def load_dicom_series_from_files(filepaths):
    dsets = []
    for p in sorted(filepaths, key=lambda x: natural_sort_key(os.path.basename(x))):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=False)
            if hasattr(ds, 'PixelData'):
                dsets.append(ds)
        except Exception as e:
            print('Skipping file', p, 'error', e)
    if not dsets:
        raise ValueError('No DICOM images found')

    def zpos(ds):
        if hasattr(ds, 'ImagePositionPatient'):
            try:
                return float(ds.ImagePositionPatient[2])
            except Exception:
                pass
        return float(getattr(ds, 'InstanceNumber', 0))

    dsets.sort(key=zpos)

    slices = []
    for ds in dsets:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        arr = arr * slope + intercept
        slices.append(arr)

    volume = np.stack(slices, axis=0)
    px = getattr(dsets[0], 'PixelSpacing', [1.0, 1.0])
    slice_thickness = float(getattr(dsets[0], 'SliceThickness', 1.0))
    spacing = (slice_thickness, float(px[0]), float(px[1]))
    return volume, spacing

def load_image_series(filepaths):
    imgs = []
    for p in sorted(filepaths, key=lambda x: natural_sort_key(os.path.basename(x))):
        im = ski_io.imread(p)
        if im.ndim == 3:
            if im.shape[2] == 4:
                im = color.rgba2rgb(im)
            im = color.rgb2gray(im)
        im = im.astype(np.float32)
        im = exposure.rescale_intensity(im, out_range=(0, 1))
        imgs.append(im)
    if not imgs:
        raise ValueError('No images found')
    shapes = [im.shape for im in imgs]
    min_shape = tuple(min(s[i] for s in shapes) for i in (0,1))
    imgs_resized = [transform.resize(im, min_shape, preserve_range=True) if im.shape!=min_shape else im for im in imgs]
    volume = np.stack(imgs_resized, axis=0)
    spacing = (1.0, 1.0, 1.0)
    return volume, spacing

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)

    files = []
    for root, _, fnames in os.walk(extract_to):  # walk through all subdirs
        for f in fnames:
            files.append(os.path.join(root, f))
    return files

# ---- Registration ----
def register_volumes(fixed_vol, moving_vol, spacing=(1.0,1.0,1.0)):
    if not HAS_SITK:
        print('SimpleITK not available — skipping registration')
        return moving_vol
    if any(dim < 4 for dim in fixed_vol.shape) or any(dim < 4 for dim in moving_vol.shape):
        print('Volume too small for registration, skipping...')
        return moving_vol
    try:
        fixed = sitk.GetImageFromArray(fixed_vol.astype(np.float32))
        moving = sitk.GetImageFromArray(moving_vol.astype(np.float32))
        fixed.SetSpacing(spacing[::-1])
        moving.SetSpacing(spacing[::-1])
        elast = sitk.ImageRegistrationMethod()
        elast.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        elast.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=50)
        elast.SetInitialTransform(
            sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY))
        elast.SetInterpolator(sitk.sitkLinear)
        out_tx = elast.Execute(fixed, moving)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(out_tx)
        out = resampler.Execute(moving)
        return sitk.GetArrayFromImage(out)
    except Exception as e:
        print(f'Registration failed: {e}')
        return moving_vol

# ---- Features / Clustering ----
def create_features(volume, spatial_weight=1.0, intensity_weight=1.0, mask=None):
    z, y, x = volume.shape
    coords = np.indices((z, y, x)).reshape(3, -1).T.astype(np.float32)
    coords *= spatial_weight
    intens = volume.reshape(-1, 1).astype(np.float32) * intensity_weight
    features = np.concatenate([coords, intens], axis=1)
    if mask is not None:
        mask_flat = mask.reshape(-1)
        features = features[mask_flat > 0]
        return features, mask_flat
    return features, None

def compute_supervoxels(volume, n_segments=1000, compactness=0.1):
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-12)
    return segmentation.slic(vol_norm, n_segments=n_segments, compactness=compactness,
                             channel_axis=None, start_label=0)

def cluster_volume(volume, algorithm='kmeans', n_clusters=5,
                   eps=0.3, min_samples=10, use_supervoxels=True,
                   supervoxel_count=2000, spatial_weight=0.01, intensity_weight=1.0):
    thresh = filters.threshold_otsu(volume.flatten())
    mask = volume > (thresh * 0.5)

    if use_supervoxels:
        sv_labels = compute_supervoxels(volume, n_segments=supervoxel_count)
        regions = measure.regionprops(sv_labels + 1, intensity_image=volume)
        feats = []
        for r in regions:
            zc, yc, xc = r.centroid
            feats.append([zc * spatial_weight, yc * spatial_weight, xc * spatial_weight, r.mean_intensity * intensity_weight])
        feats = np.array(feats, dtype=np.float32)
        scaler = StandardScaler()
        feats_scaled = scaler.fit_transform(feats)
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif algorithm == 'dbscan':
            model = DBSCAN(eps=eps, min_samples=min_samples)
        elif algorithm == 'ward':
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        elif algorithm == 'birch':
            model = Birch(n_clusters=n_clusters)
        else:
            raise ValueError('Unsupported algorithm')
        labels = model.fit_predict(feats_scaled)
        seg = np.zeros_like(sv_labels, dtype=np.int32) - 1
        for idx, r in enumerate(regions):
            seg[sv_labels == r.label] = int(labels[idx])
        return seg, mask
    else:
        X, _ = create_features(volume, spatial_weight=spatial_weight, intensity_weight=intensity_weight, mask=mask)
        Xs = StandardScaler().fit_transform(X)
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif algorithm == 'dbscan':
            model = DBSCAN(eps=eps, min_samples=min_samples)
        elif algorithm == 'ward':
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        elif algorithm == 'birch':
            model = Birch(n_clusters=n_clusters)
        else:
            raise ValueError('Unsupported algorithm')
        labs = model.fit_predict(Xs)
        seg = np.full(volume.size, -1, dtype=np.int32)
        seg[mask.reshape(-1)] = labs
        return seg.reshape(volume.shape), mask

# ---- Routes ----
@app.route('/')
def index():
    return render_template('index.html', has_sitk=HAS_SITK)


@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('file')
    saved = []
    tempdir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])

    # Save uploaded files
    for f in files:
        if f and allowed_file(f.filename):
            name = secure_filename(f.filename)
            path = os.path.join(tempdir, name)
            f.save(path)
            saved.append(path)

    # Handle ZIPs
    zip_paths = [p for p in saved if p.lower().endswith('.zip')]
    extracted = []
    for z in zip_paths:
        extracted_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        extracted.extend(extract_zip(z, extracted_dir))  # keep all files

    all_files = extracted if extracted else saved
    return jsonify({'success': True, 'files': all_files})

@app.route('/process_series', methods=['POST'])
def process_series():
    data = request.json or {}
    print("DEBUG /process_series keys:", list(data.keys()))
    print("DEBUG files count:", len(data.get("files", [])))
    files = data.get('files', [])
    algorithm = data.get('algorithm', 'kmeans')
    n_clusters = int(data.get('n_clusters', 5))
    eps = float(data.get('eps', 0.3))
    min_samples = int(data.get('min_samples', 10))
    use_supervoxels = bool(data.get('use_supervoxels', True))
    supervoxel_count = int(data.get('supervoxel_count', 2000))
    spatial_weight = float(data.get('spatial_weight', 0.01))
    intensity_weight = float(data.get('intensity_weight', 1.0))

    dcm_files = [f for f in files if f.lower().endswith('.dcm')]
    img_files = [f for f in files if not f.lower().endswith('.dcm')]

    if dcm_files:
        vol, spacing = load_dicom_series_from_files(dcm_files)
    else:
        vol, spacing = load_image_series(img_files)

    if vol.ndim == 3:
        denoised_vol = np.zeros_like(vol)
        for i in range(vol.shape[0]):
            denoised_vol[i] = denoise_bilateral(vol[i], sigma_color=0.05, sigma_spatial=1)
        vol = denoised_vol
    else:
        vol = denoise_bilateral(vol, sigma_color=0.05, sigma_spatial=1)

    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-12)
    seg, mask = cluster_volume(vol, algorithm=algorithm, n_clusters=n_clusters, eps=eps,
                               min_samples=min_samples, use_supervoxels=use_supervoxels,
                               supervoxel_count=supervoxel_count, spatial_weight=spatial_weight,
                               intensity_weight=intensity_weight)

    mid = vol.shape[0] // 2
    orig_slice = (vol[mid] * 255).astype('uint8')
    seg_slice = seg[mid]

    buf, buf2 = io.BytesIO(), io.BytesIO()
    ski_io.imsave(buf, orig_slice, plugin='pil', format_str='PNG')
    ski_io.imsave(buf2, ((seg_slice - seg_slice.min())/(np.ptp(seg_slice)+1e-12)*255).astype('uint8'),
                  plugin='pil', format_str='PNG')
    buf.seek(0); buf2.seek(0)
    orig_b64 = base64.b64encode(buf.read()).decode('utf-8')
    seg_b64 = base64.b64encode(buf2.read()).decode('utf-8')

    seg_path = os.path.join(tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER']), 'seg.npy')
    np.save(seg_path, seg)

    return jsonify({'success': True, 'orig_slice': f'data:image/png;base64,{orig_b64}',
                    'seg_slice': f'data:image/png;base64,{seg_b64}', 'seg_path': seg_path,
                    'shape': vol.shape, 'spacing': spacing})

@app.route('/process_differential', methods=['POST'])
def process_differential():
    try:
        data = request.json
        files1 = data.get('files1', [])
        files2 = data.get('files2', [])
        do_register = bool(data.get('do_register', True))
        algorithm = data.get('algorithm', 'kmeans')
        n_clusters = int(data.get('n_clusters', 5))

        dcm1 = [f for f in files1 if f.lower().endswith('.dcm')]
        dcm2 = [f for f in files2 if f.lower().endswith('.dcm')]
        v1, spacing1 = load_dicom_series_from_files(dcm1) if dcm1 else load_image_series(files1)
        v2, spacing2 = load_dicom_series_from_files(dcm2) if dcm2 else load_image_series(files2)

        min_shape = tuple(min(a,b) for a,b in zip(v1.shape, v2.shape))
        v1, v2 = v1[:min_shape[0], :min_shape[1], :min_shape[2]], v2[:min_shape[0], :min_shape[1], :min_shape[2]]

        if do_register and HAS_SITK and all(dim >= 4 for dim in v1.shape):
            v2 = register_volumes(v1, v2, spacing=spacing1)

        v1 = (v1 - v1.mean()) / (v1.std() + 1e-12)
        v2 = (v2 - v2.mean()) / (v2.std() + 1e-12)
        diff = v2 - v1

        seg, mask = cluster_volume(diff, algorithm=algorithm, n_clusters=n_clusters)
        seg_path = os.path.join(tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER']), 'diff_seg.npy')
        np.save(seg_path, seg)

        mid = diff.shape[0] // 2
        orig_slice = ((diff[mid] - diff[mid].min())/(np.ptp(diff[mid])+1e-12)*255).astype('uint8')
        seg_slice = seg[mid]

        buf, buf2 = io.BytesIO(), io.BytesIO()
        ski_io.imsave(buf, orig_slice, plugin='pil', format_str='PNG')
        ski_io.imsave(buf2, ((seg_slice - seg_slice.min())/(np.ptp(seg_slice)+1e-12)*255).astype('uint8'),
                      plugin='pil', format_str='PNG')
        buf.seek(0); buf2.seek(0)
        o_b64 = base64.b64encode(buf.read()).decode('utf-8')
        s_b64 = base64.b64encode(buf2.read()).decode('utf-8')

        return jsonify({'success': True, 'diff_slice': f'data:image/png;base64,{o_b64}',
                        'seg_slice': f'data:image/png;base64,{s_b64}', 'seg_path': seg_path,
                        'shape': diff.shape})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})



from scipy.ndimage import binary_dilation, binary_closing, label as cc_label

@app.route('/generate_3d_mesh', methods=['POST'])
def generate_3d_mesh():
    data = request.json or {}
    seg_path = data.get('seg_path')
    if not seg_path or not os.path.exists(seg_path):
        return jsonify({'success': False, 'error': 'seg_path missing or invalid'})

    try:
        seg = np.load(seg_path)

        # Light downsample only if huge (keeps browser stable); still lecture-safe.
        if max(seg.shape) > 160:
            factor = 2
            new_shape = (max(1, seg.shape[0]//factor),
                         max(1, seg.shape[1]//factor),
                         max(1, seg.shape[2]//factor))
            seg = resize(seg, new_shape, order=0, preserve_range=True, anti_aliasing=False).astype(seg.dtype)

        meshes = []
        vals, counts = np.unique(seg, return_counts=True)
        labels = [int(v) for v, c in zip(vals, counts) if v >= 0]

        # Lecture-only knobs
        MIN_VOXELS = 100        # ignore tiny specks
        DILATE_ITERS = 1        # thicken a bit (slides: dilation)
        CLOSE_ITERS  = 2        # fill small gaps (slides: closing)

        def add_mesh(label_id, verts, faces):
            meshes.append({
                'cluster_id': int(label_id),
                'x': verts[:, 0].tolist(),
                'y': verts[:, 1].tolist(),
                'z': verts[:, 2].tolist(),
                'i': faces[:, 0].tolist(),
                'j': faces[:, 1].tolist(),
                'k': faces[:, 2].tolist()
            })

        for lab in labels:
            mask = (seg == lab)
            if int(mask.sum()) < MIN_VOXELS:
                continue

            # pad borders then dilate + close (all in your lecture)
            m = np.pad(mask, 1, mode='constant', constant_values=False)
            if DILATE_ITERS > 0:
                m = binary_dilation(m, iterations=DILATE_ITERS)
            if CLOSE_ITERS > 0:
                m = binary_closing(m, iterations=CLOSE_ITERS)

            try:
                v, f, n, _ = marching_cubes(m.astype(np.float32), level=0.5)
                if len(f) > 0:
                    add_mesh(lab, v, f)
            except Exception as e:
                print(f"DEBUG: marching_cubes label {lab} failed: {e}")

        note = None

        # Global fallback: largest connected component of union (lecture: connected components)
        if not meshes:
            union = (seg >= 0)
            if union.any():
                u = np.pad(union, 1, mode='constant', constant_values=False)
                if DILATE_ITERS > 0:
                    u = binary_dilation(u, iterations=DILATE_ITERS)
                if CLOSE_ITERS > 0:
                    u = binary_closing(u, iterations=CLOSE_ITERS)

                cc, ncc = cc_label(u)
                if ncc > 0:
                    sizes = np.bincount(cc.ravel()); sizes[0] = 0
                    biggest = (cc == sizes.argmax())
                    try:
                        v, f, n, _ = marching_cubes(biggest.astype(np.float32), level=0.5)
                        if len(f) > 0:
                            add_mesh(-999, v, f)
                            note = "Largest connected region (after dilation/closing)."
                    except Exception as e:
                        print(f"DEBUG: fallback marching_cubes failed: {e}")
                        note = "No valid 3D surface; increase dilation/closing or reduce clusters."
                else:
                    note = "Union had no connected foreground."
            else:
                note = "Segmentation had no non-negative labels."

        return jsonify({
            'success': True,
            'meshes': meshes,
            'cluster_count': len(meshes),
            'unique_labels': labels,
            'note': note
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/generate_3d_from_seg', methods=['POST'])
def generate_3d_from_seg():
    data = request.json
    seg_path = data.get('seg_path')
    if not seg_path or not os.path.exists(seg_path):
        return jsonify({'error': 'seg_path missing or invalid'})
    try:
        seg = np.load(seg_path)
        return jsonify({'success': True, 'volume_data': seg.tolist(), 'shape': seg.shape})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 