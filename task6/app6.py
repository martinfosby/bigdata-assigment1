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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.measure import marching_cubes
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
    for root, _, fnames in os.walk(extract_to):
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
def create_features(volume, spatial_weight=1.0, intensity_weight=1.0, mask=None, use_pca=False, pca_components=3):
    z, y, x = volume.shape
    coords = np.indices((z, y, x)).reshape(3, -1).T.astype(np.float32)
    coords *= spatial_weight
    intens = volume.reshape(-1, 1).astype(np.float32) * intensity_weight
    features = np.concatenate([coords, intens], axis=1)
    
    if mask is not None:
        mask_flat = mask.reshape(-1)
        features = features[mask_flat > 0]
    
    if use_pca and features.shape[0] > pca_components:
        pca = PCA(n_components=min(pca_components, features.shape[1]))
        features = pca.fit_transform(features)
    
    return features, mask_flat if mask is not None else None

def compute_supervoxels(volume, n_segments=1000, compactness=0.1):
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-12)
    return segmentation.slic(vol_norm, n_segments=n_segments, compactness=compactness,
                             channel_axis=None, start_label=0)

# ... (all your imports and setup code remains the same)

def cluster_volume(volume, algorithm='kmeans', n_clusters=5,
                   eps=0.3, min_samples=10, use_supervoxels=True,
                   supervoxel_count=2000, spatial_weight=0.01, 
                   intensity_weight=1.0, use_pca=False, pca_components=3):
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
        
        if use_pca:
            pca = PCA(n_components=min(pca_components, feats.shape[1]))
            feats = pca.fit_transform(feats)
            
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
        elif algorithm == 'meanshift':
            bandwidth = estimate_bandwidth(feats_scaled, quantile=0.2, n_samples=500)
            model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        else:
            raise ValueError('Unsupported algorithm')
            
        if algorithm == 'meanshift':
            model.fit(feats_scaled)
            labels = model.labels_
            n_clusters = len(np.unique(labels))
        else:
            labels = model.fit_predict(feats_scaled)
            
        seg = np.zeros_like(sv_labels, dtype=np.int32) - 1
        for idx, r in enumerate(regions):
            seg[sv_labels == r.label] = int(labels[idx])
        return seg, mask, n_clusters
    else:
        X, mask_flat = create_features(volume, spatial_weight=spatial_weight, 
                                      intensity_weight=intensity_weight, mask=mask,
                                      use_pca=use_pca, pca_components=pca_components)
        Xs = StandardScaler().fit_transform(X)
        
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif algorithm == 'dbscan':
            model = DBSCAN(eps=eps, min_samples=min_samples)
        elif algorithm == 'ward':
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        elif algorithm == 'birch':
            model = Birch(n_clusters=n_clusters)
        elif algorithm == 'meanshift':
            bandwidth = estimate_bandwidth(Xs, quantile=0.2, n_samples=500)
            model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        else:
            raise ValueError('Unsupported algorithm')
            
        if algorithm == 'meanshift':
            model.fit(Xs)
            labs = model.labels_
            n_clusters = len(np.unique(labs))
        else:
            labs = model.fit_predict(Xs)
            
        seg = np.full(volume.size, -1, dtype=np.int32)
        seg[mask.reshape(-1)] = labs
        return seg.reshape(volume.shape), mask, n_clusters


# ---- Routes ----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})

    uploaded_files = request.files.getlist('file')
    saved_files = []

    upload_dir = os.path.join('uploads')
    os.makedirs(upload_dir, exist_ok=True)

    for f in uploaded_files:
        if f.filename == '':
            continue
        # only allow certain file extensions
        allowed_exts = ('.dcm', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
        if not f.filename.lower().endswith(allowed_exts):
            continue

        save_path = os.path.join(upload_dir, f.filename)
        f.save(save_path)
        saved_files.append(save_path)

    if not saved_files:
        return jsonify({'success': False, 'error': 'No supported files uploaded'})

    return jsonify({'success': True, 'files': saved_files})


@app.route('/process_series', methods=['POST'])
def process_series():
    data = request.json or {}
    files = data.get('files', [])
    algorithm = data.get('algorithm', 'kmeans')
    n_clusters = int(data.get('n_clusters', 5))
    eps = float(data.get('eps', 0.3))
    min_samples = int(data.get('min_samples', 10))
    use_supervoxels = bool(data.get('use_supervoxels', True))
    supervoxel_count = int(data.get('supervoxel_count', 2000))
    spatial_weight = float(data.get('spatial_weight', 0.01))
    intensity_weight = float(data.get('intensity_weight', 1.0))
    use_pca = bool(data.get('use_pca', False))
    pca_components = int(data.get('pca_components', 3))

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
    seg, mask, actual_clusters = cluster_volume(
        vol, algorithm=algorithm, n_clusters=n_clusters, eps=eps,
        min_samples=min_samples, use_supervoxels=use_supervoxels,
        supervoxel_count=supervoxel_count, spatial_weight=spatial_weight,
        intensity_weight=intensity_weight, use_pca=use_pca, pca_components=pca_components
    )

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

    return jsonify({
        'success': True, 
        'orig_slice': f'data:image/png;base64,{orig_b64}',
        'seg_slice': f'data:image/png;base64,{seg_b64}', 
        'seg_path': seg_path,
        'shape': vol.shape, 
        'spacing': spacing,
        'n_clusters': actual_clusters
    })

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

        seg, mask, actual_clusters = cluster_volume(diff, algorithm=algorithm, n_clusters=n_clusters)
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

        return jsonify({
            'success': True, 
            'diff_slice': f'data:image/png;base64,{o_b64}',
            'seg_slice': f'data:image/png;base64,{s_b64}', 
            'seg_path': seg_path,
            'shape': diff.shape,
            'n_clusters': actual_clusters
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate_3d_mesh', methods=['POST'])
def generate_3d_mesh():
    data = request.json or {}
    seg_path = data.get('seg_path')
    if not seg_path or not os.path.exists(seg_path):
        return jsonify({'success': False, 'error': 'seg_path missing or invalid'})

    try:
        seg = np.load(seg_path)
        
        # Generate random colors for each cluster (as RGB values between 0-1)
        unique_clusters = np.unique(seg)
        cluster_colors = {}
        for cid in unique_clusters:
            if cid >= 0:  # Skip background
                cluster_colors[int(cid)] = [
                    random.random(),  # R (0-1)
                    random.random(),  # G (0-1)
                    random.random(),  # B (0-1)
                    0.8  # Alpha
                ]
        
        # Create point cloud data for each cluster
        point_clouds = []
        total_points = 0
        
        for cid in unique_clusters:
            if cid < 0:  # Skip background
                continue
                
            # Get coordinates of points in this cluster
            z_coords, y_coords, x_coords = np.where(seg == cid)
            
            # Downsample if there are too many points for performance
            max_points_per_cluster = 5000
            if len(x_coords) > max_points_per_cluster:
                indices = np.random.choice(len(x_coords), max_points_per_cluster, replace=False)
                x_coords = x_coords[indices]
                y_coords = y_coords[indices]
                z_coords = z_coords[indices]
            
            # Convert to lists
            x_list = x_coords.astype(int).tolist()
            y_list = y_coords.astype(int).tolist()
            z_list = z_coords.astype(int).tolist()
            
            point_clouds.append({
                'cluster_id': int(cid),
                'x': x_list,
                'y': y_list,
                'z': z_list,
                'color': cluster_colors[int(cid)],
                'point_count': len(x_list)
            })
            total_points += len(x_list)

        return jsonify({
            'success': True,
            'point_clouds': point_clouds,
            'stats': {
                'n_clusters': len(point_clouds),
                'total_voxels': int(np.prod(seg.shape)),
                'segmented_voxels': int(np.sum(seg >= 0)),
                'total_points_displayed': total_points
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
