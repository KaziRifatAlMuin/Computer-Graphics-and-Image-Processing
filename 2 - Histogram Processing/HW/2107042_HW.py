import cv2
import numpy as np
import matplotlib.pyplot as plt

input_img = cv2.imread("boat.jpg")
reference_img = cv2.imread("power_plant.jpg")

# OpenCV reads BGR, so convert to RGB

input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
reference_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

# Convert RGB images to Lab color space

input_lab = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2LAB)
reference_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB)

# Only take the L channel

input_L = input_lab[:, :, 0]
reference_L = reference_lab[:, :, 0]

# Calculate PDFs

input_hist = np.bincount(input_L.ravel(), minlength=256)
reference_hist = np.bincount(reference_L.ravel(), minlength=256)

input_pdf = input_hist / input_L.size
reference_pdf = reference_hist / reference_L.size

# Calculate CDFs

input_cdf = np.cumsum(input_pdf)
reference_cdf = np.cumsum(reference_pdf)

# Histogram matching
# For every input intensity r:
# Find z whose reference CDF is closest to input CDF.

mapping = np.zeros(256, dtype=np.uint8)

for r in range(256):
    difference = np.abs(reference_cdf - input_cdf[r])
    z = np.argmin(difference)
    mapping[r] = z

output_L = mapping[input_L]
output_lab = input_lab.copy()
output_lab[:, :, 0] = output_L

# Back to RGB

output_rgb = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)

# Calculate output PDF and CDF

output_hist = np.bincount(output_L.ravel(), minlength=256)

output_pdf = output_hist / output_L.size
output_cdf = np.cumsum(output_pdf)

# Display results

plt.figure(figsize=(15, 10))

# Input image

plt.subplot(3, 3, 1)
plt.imshow(input_rgb)
plt.title("Input Image")
plt.axis("off")

# Input PDF

plt.subplot(3, 3, 2)
plt.plot(input_pdf, color="red")
plt.title("Source PDF")
plt.xlabel("L")
plt.ylabel("Probability")
plt.xlim(0, 255)
plt.ylim(0, max(input_pdf) * 1.1)

# Input CDF

plt.subplot(3, 3, 3)
plt.plot(input_cdf, color="red")
plt.title("Source CDF - S(r)")
plt.xlabel("L")
plt.ylabel("CDF")
plt.xlim(0, 255)
plt.ylim(0, 1)

# Reference image

plt.subplot(3, 3, 4)
plt.imshow(reference_rgb)
plt.title("Reference Image")
plt.axis("off")

# Reference PDF

plt.subplot(3, 3, 5)
plt.plot(reference_pdf, color="green")
plt.title("Reference PDF")
plt.xlabel("L")
plt.ylabel("Probability")
plt.xlim(0, 255)
plt.ylim(0, max(reference_pdf) * 1.1)

# Reference CDF

plt.subplot(3, 3, 6)
plt.plot(reference_cdf, color="green")
plt.title("Reference CDF - G(z)")
plt.xlabel("L")
plt.ylabel("CDF")
plt.xlim(0, 255)
plt.ylim(0, 1)

# Output image

plt.subplot(3, 3, 7)
plt.imshow(output_rgb)
plt.title("Output Image")
plt.axis("off")

# Output PDF

plt.subplot(3, 3, 8)
plt.plot(output_pdf, color="blue")
plt.title("Output PDF")
plt.xlabel("L")
plt.ylabel("Probability")
plt.xlim(0, 255)
plt.ylim(0, max(output_pdf) * 1.1)

# Output CDF

plt.subplot(3, 3, 9)
plt.plot(output_cdf, color="blue")
plt.title("Output CDF")
plt.xlabel("L")
plt.ylabel("CDF")
plt.xlim(0, 255)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("histogram_matching_plots.jpg", dpi=300)
plt.show()

output_save = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite("histogram_matched_output.jpg", output_save)

print("Histogram matching completed successfully.")