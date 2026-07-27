import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"Inputs\chest.png", cv2.IMREAD_GRAYSCALE)

# output = 255 - image

height = image.shape[0]
width = image.shape[1]

output = np.empty_like(image)

for i in range(height):
    for j in range(width):
        output[i, j] = 255 - image[i, j]

plt.figure(figsize=(4, 3))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap="gray", vmin=0, vmax=255)
plt.title("Negative")
plt.axis("off")

plt.tight_layout()
plt.savefig(r"Outputs\Negative\original_negative.png", dpi=300, bbox_inches="tight")
plt.show()

cv2.imwrite(r"Outputs\Negative\original.png", image)
cv2.imwrite(r"Outputs\Negative\negative.png", output)