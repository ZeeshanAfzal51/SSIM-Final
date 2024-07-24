import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def load_image(image_path):
    return cv2.imdecode(np.frombuffer(open(image_path, "rb").read(), np.uint8), 1)

def resize_image(image, target_width=None, target_height=None):
    h, w = image.shape[:2]

    if target_width is None and target_height is None:
        return image

    if target_width is None:
        r = target_height / float(h)
        dim = (int(w * r), target_height)
    else:
        r = target_width / float(w)
        dim = (target_width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def compute_ssim(image1, image2):
    # Converting images to grayscale as SSIM works only on grayscale
    grayscale1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Resize images to ensure they have the same dimensions
    if grayscale1.shape != grayscale2.shape:
        min_height = min(grayscale1.shape[0], grayscale2.shape[0])
        min_width = min(grayscale1.shape[1], grayscale2.shape[1])
        grayscale1 = grayscale1[:min_height, :min_width]
        grayscale2 = grayscale2[:min_height, :min_width]

    # Compute SSIM value between the two images
    score, _ = ssim(grayscale1, grayscale2, full=True)
    return score

def categorize_similarity(ssim_score): # Final Result Parameter
    if ssim_score == 1: # SSIM model is best for finding out if the images are a copy or not
        return "Same Images"
    elif ssim_score > 0.5: 
        return "Very Similar Images"
    else:
        return "Dissimilar Images"

# Main function
def main(image1_path, image2_path):
    if image1_path and image2_path:
        image1 = load_image(image1_path)
        image2 = load_image(image2_path)

        # Resize images for consistent comparison
        image1 = resize_image(image1, target_width=800)
        image2 = resize_image(image2, target_width=800)

        # Calculating SSIM
        ssim_score = compute_ssim(image1, image2)

        # Categorize similarity
        similarity = categorize_similarity(ssim_score)

        # Display results
        print(f"SSIM score: {ssim_score}")
        print(f"Similarity: {similarity}")

# Example usage
if __name__ == "__main__":
    image1_path = "path_to_image1.jpg"
    image2_path = "path_to_image2.jpg"
    main(image1_path, image2_path)
