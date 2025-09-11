# 2) Harnverhalt. Load the medical images for harnverhalt2 using OpenCV. Harnverhalt2.zipLast ned Harnverhalt2.zip
# Convert the images to grayscale and normalize the pixel values (0-255).
# Display the original images. Apply morphological operations of your choice to enhance your results. 
# use kmeans to cluster the images. Apply kmeans clustering to segment the medical images into different regions (e.g., tumor vs. non-tumor regions). 
# Choose a reasonable number of clusters (e.g., k=3 or k=5).
# Visualize the clustered images by colouring the segments based on the cluster labels. Improve your results using PCA-

# Reconstruct the images from the reduced PCA representation and visualize the reconstruction quality.
# Compare the reconstructed images to the original ones and discuss the trade-off between dimensionality reduction and image quality. 
# Calculate the difference images and display the results. Create a presentation video for your results or arrange a meeting with me to show your results

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
import pydicom

# ---------- CONFIG ----------
INPUT_ZIP = "Harnverhalt2.zip"
INPUT_DIR = "Harnverhalt2"
OUTPUT_DIR = "output"
K = 3
PCA_COMPONENTS = 1  # or int like 50; use float for explained variance ratio
VIDEO_FPS = 1
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "originals"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "preprocessed"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "morph"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "segmented"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "segmented-pca"), exist_ok=True)
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

# Harnverhalt. Load the medical images for harnverhalt2 using OpenCV.
def load_images_from_dir():
    exts = ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp", "*.dcm")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(INPUT_DIR, "**", e), recursive=True))
    files = sorted(files)
    
    if len(files) == 0:
        raise FileNotFoundError(f"No images found in {INPUT_DIR}.")
    
    imgs = []
    for f in files:
        if f.lower().endswith(".dcm"):
            try:
                ds = pydicom.dcmread(f)
                img = ds.pixel_array
            except Exception as e:
                print("Warning: couldn't read DICOM file", f, e)
                continue
        else:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None:
                print("Warning: couldn't read", f)
                continue
        
        imgs.append((f, img))
    
    print(f"Loaded {len(imgs)} images.")
    return imgs


# Convert the images to grayscale and normalize the pixel values (0-255).
def to_grayscale_and_normalize(img):
    '''
    Convert to gray and normalize to 0-255 uint8
    '''
    # Convert to grayscale if needed
    if len(img.shape) != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Convert to float32 for normalization
    g = gray.astype(np.float32)

    # Normalize to 0-255
    g = 255 * (g - g.min()) / (g.max() - g.min() + 1e-9)

    # convert back to uint8
    return g.astype(np.uint8)

# Display the original images. Apply morphological operations of your choice to enhance your results. 
def morphological_enhancement(gray):
    """
    Apply morphological operations to enhance tumor-like regions.
    Best practice for tumor detection is usually:
    - Median filtering to reduce noise.
    - Morphological closing to fill small holes inside bright regions (potential tumors).
    - Morphological opening to remove small bright noise.
    """
    # Denoise
    denoised = cv2.medianBlur(gray, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # Fill holes (close), then remove small noise (open)
    closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened

# use kmeans to cluster the images. 
# Apply kmeans clustering to segment the medical images into different regions (e.g., tumor vs. non-tumor regions). 
# Choose a reasonable number of clusters (e.g., k=3 or k=5).
def segment_kmeans(image_gray, k=3, use_pca=False, max_components=1):
    pixels = image_gray.reshape(-1, 1).astype(np.float32)

    # K-means parameters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    attempts = 10
    flags = cv2.KMEANS_RANDOM_CENTERS

    if use_pca:
        # Compute PCA basis
        mean, eigenvectors = cv2.PCACompute(pixels, mean=None, maxComponents=max_components)
        # Project data into PCA space
        pixels = cv2.PCAProject(pixels, mean, eigenvectors)

    # Apply k-means
    retval, labels, centers = cv2.kmeans(np.float32(pixels), k, None, criteria, attempts, flags)
 
    # Convert to image
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_gray.shape)

    if use_pca:
        return segmented_image, labels, centers, mean, eigenvectors

    return segmented_image, labels, centers

def pca_reconstruct_image(image_gray, labels, centers, mean, eigenvectors):
    # Map centers back to original space
    centers = cv2.PCABackProject(centers, mean, eigenvectors)

    # Convert to image
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_gray.shape)

    return segmented_image


def process_and_save_all():
    # ensure_unzip_or_dir()
    imgs = load_images_from_dir()
    # video_frames = []
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
        segmented_image, labels_nopca, centers_nopca = segment_kmeans(morph, k=K, use_pca=False)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "segmented", name + f"_seg_k{K}_nopca.png"), segmented_image)

        # segmentation with PCA applied to features
        segmented_image_pca, labels_pca, centers_pca, mean_pca, eigenvectors_pca = segment_kmeans(morph, k=K, use_pca=True, max_components=PCA_COMPONENTS)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "segmented-pca", name + f"_seg_k{K}_pca_com{PCA_COMPONENTS}.png"), segmented_image_pca)

        # PCA reconstruction
        reconstructed_image_pca = pca_reconstruct_image(morph, labels_pca, centers_pca, mean=mean_pca, eigenvectors=eigenvectors_pca)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "pca_recon", name + f"_recon_k{K}_pca_com{PCA_COMPONENTS}.png"), reconstructed_image_pca)

        
    #     # difference image
    #     diff = cv2.absdiff(gray, recon)
    #     # scale diff for visibility
    #     diff_vis = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    #     cv2.imwrite(os.path.join(OUTPUT_DIR, "diff", name + f"_diff.png"), diff_vis)

    #     # create montage for video: original | morph | seg(pca) | recon | diff
    #     orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     morph_rgb = cv2.cvtColor(cv2.merge([morph,morph,morph]), cv2.COLOR_BGR2RGB)
    #     seg_rgb = vis_pca
    #     recon_rgb = cv2.cvtColor(cv2.merge([recon,recon,recon]), cv2.COLOR_BGR2RGB)
    #     diff_rgb = cv2.cvtColor(cv2.merge([diff_vis,diff_vis,diff_vis]), cv2.COLOR_BGR2RGB)

    #     # resize all to same height
    #     target_h = 256
    #     def resize_keep(img_in):
    #         h0, w0 = img_in.shape[:2]
    #         scale = target_h / float(h0)
    #         return cv2.resize(img_in, (int(w0*scale), target_h))
    #     tiles = [resize_keep(x) for x in [orig_rgb, morph_rgb, seg_rgb, recon_rgb, diff_rgb]]
    #     montage = np.hstack(tiles)
    #     montage_bgr = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
    #     video_frames.append(montage_bgr)

    # # Save video
    # video_path = os.path.join(OUTPUT_DIR, "presentation.mp4")
    # print(f"Writing video to {video_path} ...")
    # writer = imageio.get_writer(video_path, fps=VIDEO_FPS)
    # for fr in video_frames:
    #     # imageio expects RGB
    #     writer.append_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    # writer.close()
    # print("Done. Outputs saved in", OUTPUT_DIR)

if __name__ == "__main__":
    process_and_save_all()
