# "MA2071 B24 Project by Tahsinur Rahim. Email: trahim@wpi.edu"
# "Data Acquisition: I will be using this image of me at the Grand Canyon, because it was a memorable trip I had the pleasure of doing with my friends in boarding school."
# "IMAGE PREPROCESSING:"
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Load the image
image_path = r"C:\Users\user\Downloads\Picture of Me at the Grand Canyon.jpg"
image0 = Image.open(image_path)
image0_np = np.array(image0)
print("Image Shape:", image0_np.shape)
plt.imshow(image0_np)
plt.axis('off')
plt.show()

# "LINEAR TRANSFORMATIONS:"
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# "identity matrix:"
T = np.array([[1, 0], [0, 1]])
height, width = image_np.shape[:2]
x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)
transformed_coords = coords @ T.T
transformed_image = np.zeros_like(image_np)

# Apply identity transformation (no change)
for i, (x, y) in enumerate(transformed_coords):
    x, y = int(round(x)), int(round(y))
    if 0 <= x < width and 0 <= y < height:
        transformed_image[y, x] = image_np[coords[i, 1], coords[i, 0]]

transformed_image_pil = Image.fromarray(transformed_image)
transformed_image_pil.show()

# "T1, T2, T3"
T1 = np.array([[0.5, 0], [0, 0.5]])  # Scaling by 0.5
T2 = np.array([[-3 / 5, 4 / 5], [4 / 5, 3 / 5]])  # Reflection through y = 2x
T3 = np.array([[3 / 5, 4 / 5], [4 / 5, -3 / 5]])  # Reflection through y = -1/2x

# "Compositions"
T = T2 @ T1
T_inv = np.linalg.inv(T)

# "Shearing when k = 0.5"
T_shear = np.array([[1, 0.5], [0, 1]])  # Shearing with k = 0.5

# Function to apply a transformation matrix
def apply_transformation(image_np, T):
    height, width = image_np.shape[:2]
    transformed_image = np.zeros_like(image_np)
    
    # Apply transformation to each pixel
    for i in range(height):
        for j in range(width):
            original_coords = np.array([j, i])  # (x, y) coordinates
            new_coords = T @ original_coords
            new_x, new_y = int(round(new_coords[0])), int(round(new_coords[1]))
            if 0 <= new_x < width and 0 <= new_y < height:
                transformed_image[new_y, new_x] = image_np[i, j]
    return transformed_image

# Apply scaling (T1)
scaled_image = apply_transformation(image_np, T1)
scaled_image_pil = Image.fromarray(scaled_image)
scaled_image_pil.show()

# Apply reflection (T2)
reflected_image = apply_transformation(image_np, T2)
reflected_image_pil = Image.fromarray(reflected_image)
reflected_image_pil.show()

# Apply the composition (T2 @ T1)
composed_image = apply_transformation(image_np, T)
composed_image_pil = Image.fromarray(composed_image)
composed_image_pil.show()

# Apply inverse transformation (T_inv)
inverse_image = apply_transformation(image_np, T_inv)
inverse_image_pil = Image.fromarray(inverse_image)
inverse_image_pil.show()

# Apply shearing transformation (T_shear)
sheared_image = apply_transformation(image_np, T_shear)
sheared_image_pil = Image.fromarray(sheared_image)
sheared_image_pil.show()

plt.imsave("scaled_image.png", scaled_image)
plt.imsave("reflected_image.png", reflected_image)
plt.imsave("composed_image.png", composed_image)
plt.imsave("inverse_image.png", inverse_image)
plt.imsave("sheared_image.png", sheared_image)







