import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import filters
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_grayscale_image(image_path):
    image = Image.open(image_path).convert("L")
    return np.array(image)


def show_img(img, width=10):
    plt.figure(figsize=(width, width / 1000 * 727))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


def erode(image, kernel, iterations=1):

    result = image.copy()
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    for _ in range(iterations):

        padded = np.pad(result, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        new_image = np.zeros_like(result)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                region = padded[i : i + kh, j : j + kw]

                # Erosion: all kernel positions must match foreground
                if np.all(region[kernel == 1] == 255):
                    new_image[i, j] = 255
        result = new_image
    return result


def dilate(image, kernel, iterations=1):

    result = image.copy()
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    for _ in range(iterations):
        padded = np.pad(result, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        new_image = np.zeros_like(result)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                region = padded[i : i + kh, j : j + kw]

                # Dilation: any kernel position matches foreground
                if np.any(region[kernel == 1] == 255):
                    new_image[i, j] = 255
        result = new_image

    return result


def morphological_open(image, kernel, iterations=1):
    eroded = erode(image, kernel, iterations)
    opened = dilate(eroded, kernel, iterations)
    return opened


def get_external_boundary_mask(binary):

    binary = (binary > 0).astype(np.uint8)
    h, w = binary.shape

    # Find external black background
    external_bg = np.zeros((h, w), dtype=bool)

    stack = []

    for x in range(w):
        if binary[0, x] == 0:
            stack.append((0, x))

        if binary[h - 1, x] == 0:
            stack.append((h - 1, x))

    for y in range(h):

        if binary[y, 0] == 0:
            stack.append((y, 0))

        if binary[y, w - 1] == 0:
            stack.append((y, w - 1))

    neighbors4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w:
            continue

        if external_bg[y, x]:
            continue

        if binary[y, x] != 0:
            continue

        external_bg[y, x] = True
        for dy, dx in neighbors4:
            stack.append((y + dy, x + dx))

    # Keep only white pixels touching external background

    boundary = np.zeros((h, w), dtype=np.uint8)

    neighbors8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0:
                continue

            for dy, dx in neighbors8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if external_bg[ny, nx]:
                        boundary[y, x] = 255
                        break
    return boundary


if __name__ == "__main__":

    binary = load_grayscale_image("binary.png")
    show_img(binary)

    kernel = np.ones((4, 4), dtype=np.uint8)

    cleaned = morphological_open(binary, kernel, iterations=2)
    show_img(cleaned)

    boundary = get_external_boundary_mask(cleaned)
    show_img(boundary)

    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(
        script_dir, "../../data/test_data/cleaning_and_boundary_extraction"
    )
    os.makedirs(test_data_dir, exist_ok=True)

    plt.imsave(
        os.path.join(test_data_dir, "get_external_boundary_mask.png"),
        boundary,
        cmap="gray",
    )
