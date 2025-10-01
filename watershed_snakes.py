import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
import pydicom

# ---------- STIER ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "Harnverhalt2")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_task5")

# Lag undermapper
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "watershed"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "snakes"), exist_ok=True)

# ---------- LAST IN DICOM-BILDE ----------
def load_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

# ---------- PREPROCESS ----------
def preprocess(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# ---------- WATERSHED ----------
def watershed_segmentation(gray, save_path):
    thresh = preprocess(gray)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    img_color[markers == -1] = [255, 0, 0]  # Røde grenser

    # --- Robust lagring ---
    try:
        success = cv2.imwrite(save_path, img_color)
        if not success:
            # fallback: bruk matplotlib hvis cv2 feiler
            plt.imsave(save_path, img_color)
            print(f"⚠️ cv2.imwrite feilet, men plt.imsave lagret {save_path}")
        else:
            print(f"✅ Lagret {save_path}")
    except Exception as e:
        print(f"❌ Kunne ikke lagre {save_path}: {e}")

    return img_color

# ---------- ACTIVE CONTOUR (SNAKES) ----------
def snakes_segmentation(gray, save_path):
    s = np.linspace(0, 2 * np.pi, 400)
    x = gray.shape[1] // 2 + (gray.shape[1] // 3) * np.cos(s)
    y = gray.shape[0] // 2 + (gray.shape[0] // 3) * np.sin(s)
    init = np.array([x, y]).T

    snake = active_contour(gray, init, alpha=0.015, beta=10, gamma=0.001)

    fig, ax = plt.subplots()
    ax.imshow(gray, cmap="gray")
    ax.plot(init[:, 0], init[:, 1], "--r", label="Startkontur")
    ax.plot(snake[:, 0], snake[:, 1], "-b", label="Resultat (Snakes)")
    ax.legend()
    plt.savefig(save_path)
    plt.close(fig)

    print(f"✅ Lagret {save_path}")
    return snake

# ---------- MAIN ----------
if __name__ == "__main__":
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".dcm")]
    if len(files) == 0:
        raise FileNotFoundError("Fant ingen .dcm-filer i Harnverhalt2-mappa!")

    # Bruk bare et lite utvalg (f.eks. første 5)
    selected_files = files[:5]

    for fname in selected_files:
        fpath = os.path.join(INPUT_DIR, fname)
        print(f"Behandler {fname} ...")

        gray = load_dicom(fpath)

        watershed_path = os.path.join(OUTPUT_DIR, "watershed", f"{fname}_watershed.png")
        watershed_segmentation(gray, watershed_path)

        snakes_path = os.path.join(OUTPUT_DIR, "snakes", f"{fname}_snakes.png")
        snakes_segmentation(gray, snakes_path)

    print(f"\nUtvalg av bilder behandlet og lagret i {OUTPUT_DIR}")
