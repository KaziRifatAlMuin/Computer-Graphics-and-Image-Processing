import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
image = cv2.imread(r"Inputs/homework_a2_b2.png", cv2.IMREAD_GRAYSCALE)

# Sharpening constant
c = 3

# Laplacian Kernel
kernel = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=np.float32)

kernel = np.flip(kernel)

def manual_convolution(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad = kh // 2

    # Zero padding
    padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=image.dtype)

    for i in range(h):
        for j in range(w):
            padded[i + pad, j + pad] = image[i, j]

    output = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            sum = 0
            for m in range(kh):
                for n in range(kw):
                    sum += padded[i + m, j + n] * kernel[m, n]
            output[i, j] = sum

    return output

# Choose One
laplacian = manual_convolution(image, kernel)
# laplacian = cv2.filter2D(image, cv2.CV_32F, kernel)

# Laplacian Sharpening
output = image.astype(np.float32) + c * laplacian

output = cv2.normalize(
    output,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)


# =====================================================
# Display
# =====================================================
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(output, cmap="gray", vmin=0, vmax=255)
plt.title("Laplacian Sharpening")
plt.axis("off")

plt.tight_layout()
plt.savefig(
    r"Outputs/Convolution/hw_a2_laplacian.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Save images
cv2.imwrite(r"Outputs/Convolution/hw_a2_original.png", image)
cv2.imwrite(r"Outputs/Convolution/hw_a2_laplacian.png", output)