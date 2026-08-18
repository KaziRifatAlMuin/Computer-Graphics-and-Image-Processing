import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"Inputs\chest.png", cv2.IMREAD_GRAYSCALE)

output_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, None, 3)
output_x_norm = cv2.normalize(output_x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

output_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, None, 3)
output_y_norm = cv2.normalize(output_y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

magnitude = cv2.magnitude(output_x, output_y)
magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

plt.figure(figsize=(4, 6))

plt.subplot(2, 2, 1)
plt.imshow(output_x_norm, cmap="gray", vmin=0, vmax=255)
plt.title("Sobel X")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(output_y_norm, cmap="gray", vmin=0, vmax=255)
plt.title("Sobel Y")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(magnitude, cmap="gray", vmin=0, vmax=255)
plt.title("Magnitude")
plt.axis("off")

plt.tight_layout()
plt.savefig(r"Outputs\Convolution\sobel_x_y.png", dpi=300, bbox_inches="tight")
plt.show()

cv2.imwrite(r"Outputs\Convolution\sobel_x.png", output_x_norm)
cv2.imwrite(r"Outputs\Convolution\sobel_y.png", output_y_norm)
cv2.imwrite(r"Outputs\Convolution\sobal_mag.png", magnitude_norm)
