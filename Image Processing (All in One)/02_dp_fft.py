import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("output", exist_ok=True)

input_img = cv2.imread("input/dp1.jpg")
reference_img = cv2.imread("input/gta6.jpg")

# OpenCV reads BGR, so convert to RGB

input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
reference_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

# Convert RGB images to Lab color space

input_lab = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2LAB)
reference_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB)

# ==========================================================
# GTA6 COLOR TRANSFER USING a AND b CHANNELS
# ==========================================================

input_a = input_lab[:, :, 1]
input_b = input_lab[:, :, 2]

reference_a = reference_lab[:, :, 1]
reference_b = reference_lab[:, :, 2]

# Calculate PDFs

input_a_hist = np.bincount(input_a.ravel(), minlength=256)
reference_a_hist = np.bincount(reference_a.ravel(), minlength=256)

input_b_hist = np.bincount(input_b.ravel(), minlength=256)
reference_b_hist = np.bincount(reference_b.ravel(), minlength=256)

input_a_pdf = input_a_hist / input_a.size
reference_a_pdf = reference_a_hist / reference_a.size

input_b_pdf = input_b_hist / input_b.size
reference_b_pdf = reference_b_hist / reference_b.size

# Calculate CDFs

input_a_cdf = np.cumsum(input_a_pdf)
reference_a_cdf = np.cumsum(reference_a_pdf)

input_b_cdf = np.cumsum(input_b_pdf)
reference_b_cdf = np.cumsum(reference_b_pdf)

# Histogram matching for a channel

mapping_a = np.zeros(256, dtype=np.uint8)

for r in range(256):
    difference = np.abs(reference_a_cdf - input_a_cdf[r])
    z = np.argmin(difference)
    mapping_a[r] = z

# Histogram matching for b channel

mapping_b = np.zeros(256, dtype=np.uint8)

for r in range(256):
    difference = np.abs(reference_b_cdf - input_b_cdf[r])
    z = np.argmin(difference)
    mapping_b[r] = z

# Apply mappings

matched_a = mapping_a[input_a]
matched_b = mapping_b[input_b]

# Blend original and matched colors

alpha = 0.50

output_a = (
    (1 - alpha) * input_a.astype(np.float32)
    + alpha * matched_a.astype(np.float32)
).astype(np.uint8)

output_b = (
    (1 - alpha) * input_b.astype(np.float32)
    + alpha * matched_b.astype(np.float32)
).astype(np.uint8)

# Keep L channel unchanged

color_lab = input_lab.copy()

color_lab[:, :, 1] = output_a
color_lab[:, :, 2] = output_b

# Convert back to RGB

color_output = cv2.cvtColor(
    color_lab,
    cv2.COLOR_LAB2RGB
)

# Save color matched image

cv2.imwrite(
    "output/01_color_matched.jpg",
    cv2.cvtColor(color_output, cv2.COLOR_RGB2BGR)
)


# ==========================================================
# FFT
# ==========================================================

gray = cv2.cvtColor(
    color_output,
    cv2.COLOR_RGB2GRAY
)

# FFT

fft = np.fft.fft2(gray)

# Shift zero frequency to center

fft_shift = np.fft.fftshift(fft)

# Magnitude spectrum

magnitude = np.abs(fft_shift)

magnitude_log = np.log1p(magnitude)

magnitude_image = cv2.normalize(
    magnitude_log,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

# Phase spectrum

phase = np.angle(fft_shift)

phase_image = cv2.normalize(
    phase,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)


# ==========================================================
# SAVE FFT MAGNITUDE AND PHASE
# ==========================================================

cv2.imwrite(
    "output/02_fft_magnitude.jpg",
    magnitude_image
)

cv2.imwrite(
    "output/03_fft_phase.jpg",
    phase_image
)


# ==========================================================
# FREQUENCY FILTER
# ==========================================================

rows, cols = gray.shape

crow = rows // 2
ccol = cols // 2

y, x = np.ogrid[:rows, :cols]

distance = np.sqrt(
    (x - ccol) ** 2 +
    (y - crow) ** 2
)

# ==========================================================
# LOW-PASS FILTER
# ==========================================================

low_radius = 35

low_pass_mask = distance <= low_radius

low_pass_fft = fft_shift * low_pass_mask

low_pass_shift = np.fft.ifftshift(
    low_pass_fft
)

low_pass = np.fft.ifft2(
    low_pass_shift
)

low_pass = np.abs(low_pass)

low_pass = cv2.normalize(
    low_pass,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imwrite(
    "output/04_low_pass.jpg",
    low_pass
)


# ==========================================================
# HIGH-PASS FILTER
# ==========================================================

high_radius = 20

high_pass_mask = distance > high_radius

high_pass_fft = fft_shift * high_pass_mask

high_pass_shift = np.fft.ifftshift(
    high_pass_fft
)

high_pass = np.fft.ifft2(
    high_pass_shift
)

high_pass = np.abs(high_pass)

# Normalize

high_pass = cv2.normalize(
    high_pass,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

# Increase contrast

high_pass = cv2.normalize(
    high_pass,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

# Apply CLAHE to make edges more visible

clahe = cv2.createCLAHE(
    clipLimit=3.0,
    tileGridSize=(8, 8)
)

high_pass = clahe.apply(
    high_pass
)

cv2.imwrite(
    "output/05_high_pass_edges.jpg",
    high_pass
)


# ==========================================================
# EDGE ENHANCEMENT
# ==========================================================

# Normalize edge image

edge_float = high_pass.astype(
    np.float32
) / 255.0

# Increase edge visibility

edge_strength = 0.65

color_float = color_output.astype(
    np.float32
)

edge_rgb = cv2.cvtColor(
    high_pass,
    cv2.COLOR_GRAY2RGB
).astype(np.float32)

enhanced_output = (
    color_float +
    edge_strength * edge_rgb
)

enhanced_output = np.clip(
    enhanced_output,
    0,
    255
).astype(np.uint8)

cv2.imwrite(
    "output/06_edge_highlighted.jpg",
    cv2.cvtColor(
        enhanced_output,
        cv2.COLOR_RGB2BGR
    )
)


# ==========================================================
# CANNY EDGE DETECTION
# ==========================================================

canny_edges = cv2.Canny(
    gray,
    80,
    180
)

cv2.imwrite(
    "output/07_canny_edges.jpg",
    canny_edges
)


# ==========================================================
# COMBINE FFT EDGES WITH COLOR IMAGE
# ==========================================================

# Use Canny edges as a strong structural guide

canny_rgb = cv2.cvtColor(
    canny_edges,
    cv2.COLOR_GRAY2RGB
)

final_output = enhanced_output.astype(
    np.float32
)

# Strong edge boost

edge_boost = 0.35

final_output += (
    canny_rgb.astype(np.float32)
    * edge_boost
)

final_output = np.clip(
    final_output,
    0,
    255
).astype(np.uint8)


# ==========================================================
# FINAL SHARPENING
# ==========================================================

blur = cv2.GaussianBlur(
    final_output,
    (0, 0),
    2
)

final_output = cv2.addWeighted(
    final_output,
    1.35,
    blur,
    -0.35,
    0
)

final_output = np.clip(
    final_output,
    0,
    255
).astype(np.uint8)


# Save final image

cv2.imwrite(
    "output/08_final_gta6_fft_output.jpg",
    cv2.cvtColor(
        final_output,
        cv2.COLOR_RGB2BGR
    )
)


# ==========================================================
# DISPLAY ALL RESULTS
# ==========================================================

plt.figure(figsize=(16, 16))

# 1. Input

plt.subplot(4, 4, 1)
plt.imshow(input_rgb)
plt.title("Input Image")
plt.axis("off")

# 2. GTA6 reference

plt.subplot(4, 4, 2)
plt.imshow(reference_rgb)
plt.title("GTA6 Reference")
plt.axis("off")

# 3. Color matched

plt.subplot(4, 4, 3)
plt.imshow(color_output)
plt.title("GTA6 Color Matched")
plt.axis("off")

# 4. FFT magnitude

plt.subplot(4, 4, 4)
plt.imshow(
    magnitude_image,
    cmap="gray"
)
plt.title("FFT Magnitude")
plt.axis("off")

# 5. FFT phase

plt.subplot(4, 4, 5)
plt.imshow(
    phase_image,
    cmap="gray"
)
plt.title("FFT Phase")
plt.axis("off")

# 6. Low pass

plt.subplot(4, 4, 6)
plt.imshow(
    low_pass,
    cmap="gray"
)
plt.title("Low-Pass Filter")
plt.axis("off")

# 7. High pass

plt.subplot(4, 4, 7)
plt.imshow(
    high_pass,
    cmap="gray"
)
plt.title("High-Pass Filter")
plt.axis("off")

# 8. Canny

plt.subplot(4, 4, 8)
plt.imshow(
    canny_edges,
    cmap="gray"
)
plt.title("Canny Edges")
plt.axis("off")

# 9. Edge highlighted

plt.subplot(4, 4, 9)
plt.imshow(
    enhanced_output
)
plt.title("FFT Edge Highlighted")
plt.axis("off")

# 10. Final

plt.subplot(4, 4, 10)
plt.imshow(
    final_output
)
plt.title("Final GTA6 + FFT")
plt.axis("off")

# 11. FFT spectrum

plt.subplot(4, 4, 11)
plt.imshow(
    magnitude_image,
    cmap="inferno"
)
plt.title("FFT Spectrum")
plt.axis("off")

# 12. Phase

plt.subplot(4, 4, 12)
plt.imshow(
    phase_image,
    cmap="gray"
)
plt.title("Phase Spectrum")
plt.axis("off")

# 13. Input a histogram

plt.subplot(4, 4, 13)
plt.plot(
    input_a_pdf,
    color="red"
)
plt.title("Input a PDF")
plt.xlabel("a")
plt.ylabel("Probability")
plt.xlim(0, 255)

# 14. Reference a histogram

plt.subplot(4, 4, 14)
plt.plot(
    reference_a_pdf,
    color="green"
)
plt.title("GTA6 a PDF")
plt.xlabel("a")
plt.ylabel("Probability")
plt.xlim(0, 255)

# 15. Input b histogram

plt.subplot(4, 4, 15)
plt.plot(
    input_b_pdf,
    color="red"
)
plt.title("Input b PDF")
plt.xlabel("b")
plt.ylabel("Probability")
plt.xlim(0, 255)

# 16. Reference b histogram

plt.subplot(4, 4, 16)
plt.plot(
    reference_b_pdf,
    color="green"
)
plt.title("GTA6 b PDF")
plt.xlabel("b")
plt.ylabel("Probability")
plt.xlim(0, 255)

plt.tight_layout()

# Save complete figure

plt.savefig(
    "output/09_all_results.jpg",
    dpi=300
)

plt.show()

print("Processing completed successfully.")
print()
print("Generated files:")
print("01_color_matched.jpg")
print("02_fft_magnitude.jpg")
print("03_fft_phase.jpg")
print("04_low_pass.jpg")
print("05_high_pass_edges.jpg")
print("06_edge_highlighted.jpg")
print("07_canny_edges.jpg")
print("08_final_gta6_fft_output.jpg")
print("09_all_results.jpg")