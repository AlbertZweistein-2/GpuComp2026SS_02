# GPU-Accelerated Detection and Classification of Jigsaw Puzzle Pieces in Digital Images
*Group 2*
*GPU Computing, Parallel Image Processing, Computer Vision*

## Group 2 Student List

| Name | GitHub Username | Task |
| :--- | :--- | :--- |
| Ananthalakshmi Vinod Iyer | ananthaiyer | **2: Segmentation**<br>File: `cleaning_and_boundary_extraction.py`<br>`erode()`<br>`dilate()`<br>`morphological_open()`<br>`get_external_boundary_mask()` |
| Jan-Hill Arganda | BeginnerCoderEurope | **4: Signal Analysis**<br>File: `contour_to_signal.py`<br>`smooth_contour()`<br>`find_triangular_peaks()`<br>(Missing: Logic to classify peaks into corners, knobs, or holes) |
| Jakob Olsacher | jacky6134 | **1: Preprocessing**<br>File: `preprocessing.py`<br>`load_grayscale_image()`<br>`create_gaussian_kernel()`<br>`gaussian_filter()`<br>`otsu_threshold()` |
| Tobias Ponesch | AlbertZweistein-2 | **5: Classification, Visualization & Integration**<br>Files: Cross-cutting<br>`show_img()`<br>(Missing: Logic to determine 4-edge labels like LLCV)<br>(Missing: Parallel drawing text/boxes)<br>(Missing: Main execution pipeline & benchmarking) |
| Philipp Sepin | p0017 | **3: Contour Extraction & Distance**<br>File: `contour_to_signal.py`<br>`trace_contour()`<br>`simplify_chain_approx()`<br>`find_contour_chain_approx_simple()`<br>`enclosing_circle_approx()`<br>`radial_signal()` |