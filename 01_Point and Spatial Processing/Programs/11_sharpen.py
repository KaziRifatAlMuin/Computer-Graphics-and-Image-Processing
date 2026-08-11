import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"Inputs\retina_2.png", cv2.IMREAD_GRAYSCALE)


kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

output = cv2.filter2D(image, cv2.CV_32F, kernel)

output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

plt.figure(figsize=(4, 3))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap="gray", vmin=0, vmax=255)
plt.title("Sharpen")
plt.axis("off")

plt.tight_layout()
plt.savefig(r"Outputs\Convolution\original_sharpen.png", dpi=300, bbox_inches="tight")
plt.show()

cv2.imwrite(r"Outputs\Convolution\original.png", image)
cv2.imwrite(r"Outputs\Convolution\sharpen.png", output)