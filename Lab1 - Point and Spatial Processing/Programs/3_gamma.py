import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"Inputs\chest.png", cv2.IMREAD_GRAYSCALE)

c = 1
gamma = 2

# output = (255 * c * (image / 255) ** gamma).astype(np.uint8)

height = image.shape[0]
width = image.shape[1]

output = np.empty_like(image)

for i in range(height):
    for j in range(width):
        r = image[i, j] / 255
        s = c * r ** gamma
        output[i, j] = (255 * s).astype(np.uint8)

plt.figure(figsize=(4, 3))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap="gray", vmin=0, vmax=255)
plt.title("Gamma")
plt.axis("off")

plt.tight_layout()
plt.savefig(r"Outputs\Gamma\original_gamma.png", dpi=300, bbox_inches="tight")
plt.show()

cv2.imwrite(r"Outputs\Gamma\original.png", image)
cv2.imwrite(r"Outputs\Gamma\gamma.png", output)