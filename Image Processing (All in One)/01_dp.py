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

# Take a and b channels

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

# Apply histogram mappings

matched_a = mapping_a[input_a]
matched_b = mapping_b[input_b]

# Blend original and matched channels

alpha = 0.5

output_a = (
    (1 - alpha) * input_a.astype(np.float32)
    + alpha * matched_a.astype(np.float32)
).astype(np.uint8)

output_b = (
    (1 - alpha) * input_b.astype(np.float32)
    + alpha * matched_b.astype(np.float32)
).astype(np.uint8)

# Keep L channel unchanged

output_lab = input_lab.copy()

output_lab[:, :, 1] = output_a
output_lab[:, :, 2] = output_b

# Convert Lab back to RGB

output_rgb = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)

# Calculate output PDFs

output_a_hist = np.bincount(output_a.ravel(), minlength=256)
output_b_hist = np.bincount(output_b.ravel(), minlength=256)

output_a_pdf = output_a_hist / output_a.size
output_b_pdf = output_b_hist / output_b.size

# Calculate output CDFs

output_a_cdf = np.cumsum(output_a_pdf)
output_b_cdf = np.cumsum(output_b_pdf)

# Display results

plt.figure(figsize=(15, 12))

# Input image

plt.subplot(4, 3, 1)
plt.imshow(input_rgb)
plt.title("Input Image")
plt.axis("off")

# Input a PDF

plt.subplot(4, 3, 2)
plt.plot(input_a_pdf, color="red")
plt.title("Input a PDF")
plt.xlabel("a")
plt.ylabel("Probability")
plt.xlim(0, 255)
plt.ylim(0, max(input_a_pdf) * 1.1)

# Input b PDF

plt.subplot(4, 3, 3)
plt.plot(input_b_pdf, color="red")
plt.title("Input b PDF")
plt.xlabel("b")
plt.ylabel("Probability")
plt.xlim(0, 255)
plt.ylim(0, max(input_b_pdf) * 1.1)

# Reference image

plt.subplot(4, 3, 4)
plt.imshow(reference_rgb)
plt.title("Reference Image")
plt.axis("off")

# Reference a PDF

plt.subplot(4, 3, 5)
plt.plot(reference_a_pdf, color="green")
plt.title("Reference a PDF")
plt.xlabel("a")
plt.ylabel("Probability")
plt.xlim(0, 255)
plt.ylim(0, max(reference_a_pdf) * 1.1)

# Reference b PDF

plt.subplot(4, 3, 6)
plt.plot(reference_b_pdf, color="green")
plt.title("Reference b PDF")
plt.xlabel("b")
plt.ylabel("Probability")
plt.xlim(0, 255)
plt.ylim(0, max(reference_b_pdf) * 1.1)

# Output image

plt.subplot(4, 3, 7)
plt.imshow(output_rgb)
plt.title(f"Blended Output (α = {alpha})")
plt.axis("off")

# Output a PDF

plt.subplot(4, 3, 8)
plt.plot(output_a_pdf, color="blue")
plt.title("Output a PDF")
plt.xlabel("a")
plt.ylabel("Probability")
plt.xlim(0, 255)
plt.ylim(0, max(output_a_pdf) * 1.1)

# Output b PDF

plt.subplot(4, 3, 9)
plt.plot(output_b_pdf, color="blue")
plt.title("Output b PDF")
plt.xlabel("b")
plt.ylabel("Probability")
plt.xlim(0, 255)
plt.ylim(0, max(output_b_pdf) * 1.1)

# a CDF comparison

plt.subplot(4, 3, 10)
plt.plot(input_a_cdf, color="red", label="Input")
plt.plot(reference_a_cdf, color="green", label="Reference")
plt.plot(output_a_cdf, color="blue", label="Output")
plt.title("a CDF Comparison")
plt.xlabel("a")
plt.ylabel("CDF")
plt.xlim(0, 255)
plt.ylim(0, 1)
plt.legend()

# b CDF comparison

plt.subplot(4, 3, 11)
plt.plot(input_b_cdf, color="red", label="Input")
plt.plot(reference_b_cdf, color="green", label="Reference")
plt.plot(output_b_cdf, color="blue", label="Output")
plt.title("b CDF Comparison")
plt.xlabel("b")
plt.ylabel("CDF")
plt.xlim(0, 255)
plt.ylim(0, 1)
plt.legend()

# Output Lab image

plt.subplot(4, 3, 12)
plt.imshow(output_lab)
plt.title("Output Lab")
plt.axis("off")

plt.tight_layout()

# Save plots

plt.savefig("output/dp1_matching_plots.jpg", dpi=300)

# Save processed image

output_save = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

cv2.imwrite("output/dp1_output.jpg", output_save)

plt.show()

print("Blended histogram matching completed successfully.")
print(f"Blend factor (alpha): {alpha}")