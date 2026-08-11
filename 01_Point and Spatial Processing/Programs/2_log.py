import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"Inputs\chest.png", cv2.IMREAD_GRAYSCALE)

c = 30

# output = (c * np.log(1 + image.astype(np.float32))).astype(np.uint8)

height = image.shape[0]
width = image.shape[1]

output = np.empty_like(image)

for i in range(height):
    for j in range(width):
        r = image[i, j].astype(np.float32)
        s = c * np.log(1 + r)
        output[i, j] = s.astype(np.uint8)

plt.figure(figsize=(4, 3))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap="gray", vmin=0, vmax=255)
plt.title("Log")
plt.axis("off")

plt.tight_layout()
plt.savefig(r"Outputs\Log\original_log.png", dpi=300, bbox_inches="tight")
plt.show()

cv2.imwrite(r"Outputs\Log\original.png", image)
cv2.imwrite(r"Outputs\Log\log.png", output)