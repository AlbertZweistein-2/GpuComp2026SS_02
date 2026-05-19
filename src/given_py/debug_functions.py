## To run: python src/given_py/debug_functions.py
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

DEBUG_DIR = Path("debug_images")
DEBUG_DIR.mkdir(exist_ok=True)

sys.path.append(str(Path(__file__).parent))

from preprocessing import load_grayscale_image, gaussian_filter, otsu_threshold
from cleaning_and_boundary_extraction import (
    erode,
    dilate,
    morphological_open,
    get_external_boundary_mask,
)

from contour_to_signal import (
    find_contour_chain_approx_simple,
    smooth_contour,
    enclosing_circle_approx,
    radial_signal,
    find_triangular_peaks,
)

def describe(name, arr):
    print(f"\n{name}")
    print("type:", type(arr))
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    print("min/max:", arr.min(), arr.max())
    print("unique values (first 20):", np.unique(arr)[:20])

def save_img(name, arr):
    path = DEBUG_DIR / f"{name}.png"
    plt.imsave(path, arr, cmap="gray")
    print(f"saved: {path}")


image_path = "data/single.JPG"

# -------------------------------
# preprocessing.py functions
# -------------------------------

image = load_grayscale_image(image_path)
describe("image", image)
save_img("debug_01_image", image)

blurred = gaussian_filter(image, kernel_size=5, sigma=1)
describe("blurred", blurred)
save_img("debug_02_blurred", blurred)

binary, threshold = otsu_threshold(blurred)
print("\nthreshold:", threshold)
describe("binary", binary)
save_img("debug_03_binary", binary)

kernel = np.ones((4, 4), dtype=np.uint8)

# -------------------------------
# cleaning_and_coundary_extraction.py functions
# -------------------------------

eroded = erode(binary, kernel, iterations=1)
describe("eroded", eroded)
save_img("debug_04_eroded", eroded)

dilated = dilate(binary, kernel, iterations=1)
describe("dilated", dilated)
save_img("debug_05_dilated", dilated)

cleaned = morphological_open(binary, kernel, iterations=2)
describe("cleaned", cleaned)
save_img("debug_06_cleaned", cleaned)

boundary = get_external_boundary_mask(cleaned)
describe("boundary", boundary)
save_img("debug_07_boundary", boundary)

# -------------------------------
# contour_to_signal.py functions
# -------------------------------

contour = find_contour_chain_approx_simple(boundary)
print("\ncontour")
print("type:", type(contour))
print("shape:", contour.shape)
print("dtype:", contour.dtype)
print("first 10 points:", contour[:10])

points = [(int(x), int(y)) for x, y in contour]

smooth_points = smooth_contour(points, window=5)
print("\nsmooth_points")
print("type:", type(smooth_points))
print("length:", len(smooth_points))
print("first 10 points:", smooth_points[:10])

center, radius = enclosing_circle_approx(smooth_points)
print("\ncenter/radius")
print("center:", center)
print("radius:", radius)

signal = radial_signal(smooth_points, center)
print("\nsignal")
print("type:", type(signal))
print("length:", len(signal))
print("first 10 values:", signal[:10])
print("min/max:", min(signal), max(signal))

peaks_idx, peaks_val, smooth = find_triangular_peaks(
    signal,
    min_distance=5,
    min_prominence=5,
    smooth_window=2,
    min_sharpness=20
)

print("\npeaks")
print("peaks_idx type:", type(peaks_idx))
print("peaks_idx length:", len(peaks_idx))
print("peaks_idx:", peaks_idx)

print("peaks_val length:", len(peaks_val))
print("peaks_val:", peaks_val)

print("smooth signal type:", type(smooth))
print("smooth signal shape:", smooth.shape)
print("smooth signal dtype:", smooth.dtype)


# --------------------------------
# contour visualization
# --------------------------------

fig, ax = plt.subplots(figsize=(10, 10))

display_boundary = boundary.copy()

if display_boundary.max() <= 1:
    display_boundary = display_boundary * 255

ax.imshow(display_boundary, cmap="gray", vmin=0, vmax=255)

# Original contour
xs = [p[0] for p in points]
ys = [p[1] for p in points]
ax.plot(xs, ys, linewidth=1)

# Smoothed contour
xs_s = [p[0] for p in smooth_points]
ys_s = [p[1] for p in smooth_points]
ax.plot(xs_s, ys_s, linewidth=2)

# Circle
circle = plt.Circle(center, radius, fill=False, linewidth=2)
ax.add_patch(circle)

# Center
ax.scatter(center[0], center[1], s=30)

ax.axis("off")

plt.savefig(DEBUG_DIR / "contour_circle.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=300)

plt.close()


# --------------------------------
# radial signal plot
# --------------------------------

plt.figure(figsize=(12,4))

plt.plot(signal)

plt.title("Radial Signal")
plt.xlabel("Contour Index")
plt.ylabel("Distance")

plt.grid(True)

plt.savefig(DEBUG_DIR / "radial_signal.png",
            bbox_inches="tight",
            dpi=300)

plt.close()

# --------------------------------
# peaks plot
# --------------------------------

plt.figure(figsize=(14,5))

plt.plot(signal)

plt.scatter(
    peaks_idx,
    peaks_val,
    marker='o'
)

plt.grid(True)

plt.savefig(DEBUG_DIR / "signal_peaks.png",
            bbox_inches="tight",
            dpi=300)

plt.close()

# --------------------------------
# contour + peaks overlay
# --------------------------------

fig, ax = plt.subplots(figsize=(10,10))

ax.imshow(display_boundary, cmap="gray", vmin=0, vmax=255)

ax.plot(xs_s, ys_s, linewidth=2)

circle = plt.Circle(center, radius, fill=False, linewidth=2)
ax.add_patch(circle)

ax.scatter(center[0], center[1], s=30)

peak_points = [smooth_points[i] for i in peaks_idx]

px = [p[0] for p in peak_points]
py = [p[1] for p in peak_points]

ax.scatter(px, py, s=80)

ax.axis("off")

plt.savefig(DEBUG_DIR / "contour_circle_peaks.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=300)

plt.close()