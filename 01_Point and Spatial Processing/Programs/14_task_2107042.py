import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"Inputs\retina_2.png", cv2.IMREAD_GRAYSCALE)


kernelx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

kernely = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

cv2.rotate(kernelx, cv2.ROTATE_180)
cv2.rotate(kernely, cv2.ROTATE_180)


x = cv2.filter2D(image, cv2.CV_32F, kernelx)
y = cv2.filter2D(image, cv2.CV_32F, kernely)
magnitude = cv2.magnitude(x, y)

outputx = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
outputy = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
outputm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

plt.figure(figsize=(4, 3))

plt.subplot(2, 2, 1)
plt.imshow(outputx, cmap="gray", vmin=0, vmax=255)
plt.title("Sobel X")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(outputy, cmap="gray", vmin=0, vmax=255)
plt.title("Sobel Y")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(outputm, cmap="gray", vmin=0, vmax=255)
plt.title("Magnitude")
plt.axis("off")

plt.tight_layout()
plt.savefig(r"Outputs\Convolution\task.png", dpi=300, bbox_inches="tight")
plt.show()

cv2.imwrite(r"Outputs\Convolution\original.png", image)
cv2.imwrite(r"Outputs\Convolution\sobelx.png", outputx)
cv2.imwrite(r"Outputs\Convolution\sobely.png", outputy)

c = 30

# output = (c * np.log(1 + image.astype(np.float32))).astype(np.uint8)

# height = image.shape[0]
# width = image.shape[1]

# output = np.empty_like(image)

# for i in range(height):
#     for j in range(width):
#         r = image[i, j].astype(np.float32)
#         s = c * np.log(1 + r)
#         output[i, j] = s.astype(np.uint8)

# plt.subplot(1, 2, 1)
# plt.imshow(magnitude, cmap="gray", vmin=0, vmax=255)
# plt.title("Original")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(output, cmap="gray", vmin=0, vmax=255)
# plt.title("Log")
# plt.axis("off")

# plt.tight_layout()
# plt.savefig(r"Outputs\Log\magnitude_log.png", dpi=300, bbox_inches="tight")
# plt.show()