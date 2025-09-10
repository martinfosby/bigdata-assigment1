import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
import os
import glob
import pydicom

# ---------- LAST IN DICOM-BILDE ----------
def load_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    gray = img.astype(np.uint8)
    return img, gray

# ---------- PREPROCESS ----------
def preprocess(gray):
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# ---------- WATERSHED ----------
def watershed_segmentation(gray, thresh):
    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0

    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    img_color[markers == -1] = [255, 0, 0]  # røde grenser

    return img_color

# ---------- ACTIVE CONTOUR (SNAKES) ----------
def snakes_segmentation(gray):
    s = np.linspace(0, 2*np.pi, 400)
    x = gray.shape[1]//2 + (gray.shape[1]//3)*np.cos(s)
    y = gray.shape[0]//2 + (gray.shape[0]//3)*np.sin(s)
    init = np.array([x, y]).T

    snake = active_contour(gray, init, alpha=0.015, beta=10, gamma=0.001)

    fig, ax = plt.subplots()
    ax.imshow(gray, cmap="gray")
    ax.plot(init[:, 0], init[:, 1], '--r', label="Startkontur")
    ax.plot(snake[:, 0], snake[:, 1], '-b', label="Resultat (Snakes)")
    ax.legend()
    plt.show()

    return snake

# ---------- MAIN ----------
if __name__ == "__main__":
    # Finn base path der scriptet ligger
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "Harnverhalt2")

    # Let etter DICOM-filer
    files = glob.glob(os.path.join(data_path, "*.dcm"))
    if len(files) == 0:
        raise FileNotFoundError(f"Fant ingen .dcm-filer i {data_path}!")

    first_file = files[0]
    print(f"Bruker fil: {first_file}")

    img, gray = load_dicom(first_file)

    # Preprocess
    thresh = preprocess(gray)

    # Kjør watershed
    result_ws = watershed_segmentation(gray, thresh)

    # Kjør active contour (snakes)
    result_snake = snakes_segmentation(gray)

    # Vis resultater
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1); plt.imshow(gray, cmap="gray"); plt.title("Original (grayscale)")
    plt.subplot(1, 3, 2); plt.imshow(result_ws); plt.title("Watershed result")
    plt.subplot(1, 3, 3); plt.imshow(gray, cmap="gray"); plt.plot(result_snake[:, 0], result_snake[:, 1], '-b'); plt.title("Active Contour (Snakes)")
    plt.show()
