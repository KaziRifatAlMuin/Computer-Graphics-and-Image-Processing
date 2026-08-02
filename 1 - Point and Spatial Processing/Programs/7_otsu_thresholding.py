import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"Inputs\chest.png", cv2.IMREAD_GRAYSCALE)

threshold, output = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(4, 3))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap="gray", vmin=0, vmax=255)
plt.title("Otsu")
plt.axis("off")

plt.tight_layout()
plt.savefig(r"Outputs\Otsu\original_otsu.png", dpi=300, bbox_inches="tight")
plt.show()

cv2.imwrite(r"Outputs\Otsu\original.png", image)
cv2.imwrite(r"Outputs\Otsu\otsu.png", output)