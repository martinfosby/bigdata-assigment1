"""
harnverhalt_pipeline.py
Full pipeline:
- Les bilder fra Harnverhalt2.zip eller mappe Harnverhalt2/
- Konverter til gråskala og normaliser (0-255)
- Vis originaler (lagrer også til disk)
- Påfør morfologisk filtrering (open/close/median) for å forbedre
- Segmenter med KMeans (uten PCA og med PCA)
- Visualiser segmentene ved å fargelegge cluster-labels
- Bruk PCA for dimensjonsreduksjon på bildedata og rekonstruer
- Lag forskjellsbilde (abs(original - reconstructed))
- Lag en presentasjonsvideo som viser original, segmentert, rekonstruert og diff
"""

import os
import zipfile
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import imageio

# ---------- CONFIG ----------
INPUT_ZIP = "Harnverhalt2.zip"
INPUT_DIR = "Harnverhalt2"
OUTPUT_DIR = "output"
K = 3  # prøv 3 eller 5
PCA_COMPONENTS = 0.95  # or int like 50; use float for explained variance ratio
VIDEO_FPS = 1
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "originals"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "preprocessed"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "morph"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "segmented"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "pca_recon"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "diff"), exist_ok=True)

def ensure_unzip_or_dir():
    # If zip exists, extract it
    if os.path.exists(INPUT_ZIP):
        print(f"Extracting {INPUT_ZIP}...")
        with zipfile.ZipFile(INPUT_ZIP, 'r') as z:
            z.extractall(INPUT_DIR)
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"Could not find {INPUT_DIR} folder nor {INPUT_ZIP} zip. Please provide files.")

def load_images_from_dir():
    exts = ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(INPUT_DIR, "**", e), recursive=True))
    files = sorted(files)
    if len(files)==0:
        raise FileNotFoundError(f"No images found in {INPUT_DIR}.")
    imgs = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            print("Warning: couldn't read", f)
            continue
        imgs.append((f, img))
    print(f"Loaded {len(imgs)} images.")
    return imgs

def to_grayscale_and_normalize(img):
    # Convert to gray and normalize to 0-255 uint8
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # scale to 0-255 if not already
    g = gray.astype(np.float32)
    g = 255 * (g - g.min()) / (g.max() - g.min() + 1e-9)
    return g.astype(np.uint8)

def morphological_enhancement(gray):
    # median denoise + morphological open then close to remove small noise and fill holes
    den = cv2.medianBlur(gray, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opened = cv2.morphologyEx(den, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed

def segment_kmeans(image_gray, k=3, use_pca=False, pca_model=None):
    # image_gray: 2D array uint8
    h,w = image_gray.shape
    # feature vector: intensity + optionally XY coordinates (helps spatial coherence)
    X = image_gray.reshape(-1,1).astype(np.float32)
    # add spatial coordinates normalized
    yy, xx = np.indices((h,w))
    coords = np.stack([xx.reshape(-1)/w, yy.reshape(-1)/h], axis=1).astype(np.float32)
    features = np.concatenate([X, coords], axis=1)
    # Optionally apply PCA to features (for clustering)
    if use_pca:
        if pca_model is None:
            pca = PCA(n_components=PCA_COMPONENTS, svd_solver='full')
            features_scaled = StandardScaler().fit_transform(features)
            features_p = pca.fit_transform(features_scaled)
            pca_model = (pca, StandardScaler())
        else:
            pca, scaler = pca_model
            features_p = pca.transform(scaler.transform(features))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features_p)
    else:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features_scaled)
        pca_model = (None, scaler)
    labels_img = labels.reshape(h,w)
    return labels_img, pca_model

def colorize_labels(labels_img):
    # produce color image from label map
    h,w = labels_img.shape
    labels = np.unique(labels_img)
    out = np.zeros((h,w,3), dtype=np.uint8)
    cmap = plt.cm.get_cmap("tab10", len(labels))
    for i,lab in enumerate(labels):
        color = np.array([int(255*x) for x in cmap(i)[:3]], dtype=np.uint8)
        out[labels_img==lab] = color
    return out

def pca_reconstruct_image(image_gray, n_components=PCA_COMPONENTS):
    # Flatten image as pixels (h*w) x 1 and do PCA reconstruction across pixels features.
    # Here more sensible is to break into patches or treat entire image as vector.
    # We'll treat rows as samples, columns as features -> this keeps local structure.
    imgf = image_gray.astype(np.float32) / 255.0
    h,w = imgf.shape
    # center rows
    pca = PCA(n_components=n_components, svd_solver='full')
    X = imgf.copy()
    # either do PCA on rows or on flattened image. We'll use flattened approach:
    Xflat = X.reshape(-1,1)  # (h*w, 1) trivial for grayscale -> PCA won't compress. So better to use patches.
    # Use small patches (e.g., 8x8) to get meaningful PCA compression.
    patch_size = 8
    patches = []
    coords = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            p = imgf[y:y+patch_size, x:x+patch_size].reshape(-1)
            patches.append(p)
            coords.append((y,x))
    patches = np.array(patches)  # (n_patches, patch_size*patch_size)
    pca = PCA(n_components=n_components, svd_solver='full')
    patches_mean = patches.mean(axis=0)
    patches_centered = patches - patches_mean
    pca_fit = pca.fit(patches_centered)
    proj = pca_fit.transform(patches_centered)
    recon = pca_fit.inverse_transform(proj) + patches_mean
    # Reconstruct image from patches (average overlaps if any)
    recon_img = np.zeros_like(imgf)
    weight = np.zeros_like(imgf)
    idx = 0
    for (y,x) in coords:
        p = recon[idx].reshape(patch_size, patch_size)
        recon_img[y:y+patch_size, x:x+patch_size] += p
        weight[y:y+patch_size, x:x+patch_size] += 1.0
        idx += 1
    # for remaining edge areas not covered by patches, copy original
    mask = weight > 0
    recon_img[mask] = recon_img[mask] / weight[mask]
    recon_img[~mask] = imgf[~mask]
    recon_img = np.clip(recon_img*255.0, 0, 255).astype(np.uint8)
    return recon_img, pca_fit

def process_and_save_all():
    ensure_unzip_or_dir()
    imgs = load_images_from_dir()
    video_frames = []
    for path, img in tqdm(imgs, desc="Processing images"):
        name = os.path.splitext(os.path.basename(path))[0]
        # save original (converted BGR -> RGB for matplotlib)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "originals", name + "_orig.png"), img)

        # preprocess: gray + normalize
        gray = to_grayscale_and_normalize(img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "preprocessed", name + "_gray.png"), gray)

        # morphological enhancement
        morph = morphological_enhancement(gray)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "morph", name + "_morph.png"), morph)

        # segmentation without PCA
        labels_no_pca, model_no = segment_kmeans(morph, k=K, use_pca=False)
        vis_no_pca = colorize_labels(labels_no_pca)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "segmented", name + f"_seg_k{K}_nopca.png"), vis_no_pca[:,:,::-1])  # BGR

        # segmentation with PCA applied to features
        labels_pca, pca_model = segment_kmeans(morph, k=K, use_pca=True, pca_model=None)
        vis_pca = colorize_labels(labels_pca)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "segmented", name + f"_seg_k{K}_pca.png"), vis_pca[:,:,::-1])

        # PCA reconstruction
        recon, pca_fit = pca_reconstruct_image(morph, n_components=PCA_COMPONENTS)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "pca_recon", name + f"_recon_pc{PCA_COMPONENTS}.png"), recon)

        # difference image
        diff = cv2.absdiff(gray, recon)
        # scale diff for visibility
        diff_vis = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "diff", name + f"_diff.png"), diff_vis)

        # create montage for video: original | morph | seg(pca) | recon | diff
        orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        morph_rgb = cv2.cvtColor(cv2.merge([morph,morph,morph]), cv2.COLOR_BGR2RGB)
        seg_rgb = vis_pca
        recon_rgb = cv2.cvtColor(cv2.merge([recon,recon,recon]), cv2.COLOR_BGR2RGB)
        diff_rgb = cv2.cvtColor(cv2.merge([diff_vis,diff_vis,diff_vis]), cv2.COLOR_BGR2RGB)

        # resize all to same height
        target_h = 256
        def resize_keep(img_in):
            h0, w0 = img_in.shape[:2]
            scale = target_h / float(h0)
            return cv2.resize(img_in, (int(w0*scale), target_h))
        tiles = [resize_keep(x) for x in [orig_rgb, morph_rgb, seg_rgb, recon_rgb, diff_rgb]]
        montage = np.hstack(tiles)
        montage_bgr = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
        video_frames.append(montage_bgr)

    # Save video
    video_path = os.path.join(OUTPUT_DIR, "presentation.mp4")
    print(f"Writing video to {video_path} ...")
    writer = imageio.get_writer(video_path, fps=VIDEO_FPS)
    for fr in video_frames:
        # imageio expects RGB
        writer.append_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    writer.close()
    print("Done. Outputs saved in", OUTPUT_DIR)

if __name__ == "__main__":
    process_and_save_all()
