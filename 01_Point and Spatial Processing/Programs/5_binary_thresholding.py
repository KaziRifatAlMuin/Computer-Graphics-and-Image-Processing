import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"Inputs\chest.png", cv2.IMREAD_GRAYSCALE)

t = 130

threshold, output = cv2.threshold(image, t, 255, cv2.THRESH_BINARY)

# height = image.shape[0]
# width = image.shape[1]

# output = np.empty_like(image)

# for i in range(height):
#     for j in range(width):
#         if (image[i, j] >= t):
#             output[i, j] = 255
#         else:
#             output[i, j] = 0

plt.figure(figsize=(4, 3))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap="gray", vmin=0, vmax=255)
plt.title("Binary")
plt.axis("off")

plt.tight_layout()
plt.savefig(r"Outputs\Binary\original_binary.png", dpi=300, bbox_inches="tight")
plt.show()

cv2.imwrite(r"Outputs\Binary\original.png", image)
cv2.imwrite(r"Outputs\Binary\binary.png", output)