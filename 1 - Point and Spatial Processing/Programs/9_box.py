import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"Inputs\chest.png", cv2.IMREAD_GRAYSCALE)

output = cv2.blur(image, (25, 25))

plt.figure(figsize=(4, 3))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap="gray", vmin=0, vmax=255)
plt.title("Box")
plt.axis("off")

plt.tight_layout()
plt.savefig(r"Outputs\Convolution\original_box.png", dpi=300, bbox_inches="tight")
plt.show()

cv2.imwrite(r"Outputs\Convolution\original.png", image)
cv2.imwrite(r"Outputs\Convolution\box.png", output)