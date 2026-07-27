import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"Inputs\chest.png", cv2.IMREAD_GRAYSCALE)

height = image.shape[0]
width = image.shape[1]

t = np.mean(image)
delta = 1

while True:
    g1 = image[image > t]
    g2 = image[image <= t]

    mu1 = np.mean(g1)
    mu2 = np.mean(g2)

    t_new = (mu1 + mu2) / 2

    if (np.abs(t_new - t) < delta):
        break

    t = t_new

t = t.astype(np.uint8)

output = np.empty_like(image)

for i in range(height):
    for j in range(width):
        if (image[i, j] >= t):
            output[i, j] = 255
        else:
            output[i, j] = 0

plt.figure(figsize=(4, 3))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap="gray", vmin=0, vmax=255)
plt.title(f"Isodata, t = {t}")
plt.axis("off")

plt.tight_layout()
plt.savefig(r"Outputs\Isodata\original_isodata.png", dpi=300, bbox_inches="tight")
plt.show()

cv2.imwrite(r"Outputs\Isodata\original.png", image)
cv2.imwrite(r"Outputs\Isodata\isodata.png", output)