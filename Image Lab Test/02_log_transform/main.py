import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"02_log_transform/girl.png", cv2.IMREAD_GRAYSCALE)

height = image.shape[0]
width = image.shape[1]

c = 30

output = np.empty_like(image)

for i in range(height):
    for j in range(width):
        r = image[i, j].astype(np.float32)
        s = c * np.log(1 + r)
        output[i,j] = s.astype(np.uint8)

plt.figure(figsize=(4,3))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap = "gray", vmin = 0, vmax = 255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap = "gray", vmin = 0, vmax = 255)
plt.title("Log")
plt.axis("off")

plt.tight_layout()
plt.savefig(r"02_log_transform/log_transform.png", dpi=300, bbox_inches="tight")
plt.show()

cv2.imwrite(r"02_log_transform/original.png", image)
cv2.imwrite(r"02_log_transform/log.png", output)
