import os
import numpy as np
import cv2
import torch
from torchvision.models.segmentation import deeplabv3_resnet50,DeepLabV3_ResNet50_Weights
from torchvision import transforms
import dlib
import concurrent.futures
import argparse

def make_deeplab(device="cuda"):
    deeplab = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).to(device)
    deeplab.eval()
    return deeplab

def apply_deeplab(deeplab, img, device):
    # TODO: Find out WTF is going on with this GPT-3 generated code
    deeplab_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    input_tensor = deeplab_preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = deeplab(input_batch.to(device))['out'][0]
    output_predictions = output.argmax(0).cpu().numpy()
    return (output_predictions == 15) # LABEL_NAMES = np.asarray([ 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv' ])

def detect_faces_in_image(image):
    detector = dlib.get_frontal_face_detector()
    faces = detector(image, 0)
    largest_area = 0
    largest_face = None
    for face in faces:
        # Find the largest face by area
        x, y = face.left(), face.top()
        w, h = face.right() - face.left(), face.bottom() - face.top()
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_face = (x, y, w, h)
    return largest_face

def detect_face(images):
    print("Detecting faces...")
    largest_faces = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(detect_faces_in_image, images)
        for result in results:
            largest_faces.append(result)
    return largest_faces

def preprocess_image(image):
    h, w = image.shape[:2]
    size = max(h, w)
    pad_color = [255, 255, 255]  # RGB color for padding
    dw = (size - w) // 2
    dh = (size - h) // 2
    pad = cv2.copyMakeBorder(image, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=pad_color)
    resized = cv2.resize(pad, (2048, 2048), interpolation=cv2.INTER_LANCZOS4)
    return resized

def preprocess_to_square(images):
    print("Preprocessing images...")
    processed_images = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(preprocess_image, image) for image in images]

        for future in concurrent.futures.as_completed(futures):
            processed_images.append(future.result())

    return processed_images

def load_images_from_directory(directory):
    print("Loading images...")
    images=[]
    for file in os.listdir(directory):
        filename=os.path.join(directory,file)
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    return images

def main(*args, **kwargs):
    # Parse arguments
    # Args: --image_dir <path to input directory> --out_dir <path to output directory>
    parser = argparse.ArgumentParser(description='Preprocess images for masking')
    parser.add_argument('--image_dir', type=str, required=True, help='path to input directory')
    parser.add_argument('--out_dir', type=str, required=True, help='path to output directory')
    # Take the arguments from the command line to IMAGE_DIR and OUT_DIR
    args = parser.parse_args()
    IMAGE_DIR = args.image_dir
    OUT_DIR = args.out_dir
    
    # Load and process each image in the input directory
    counter_has_face=0
    images=load_images_from_directory(IMAGE_DIR)
    images=preprocess_to_square(images)
    largest_faces = detect_face(images)
    
    # Crop the image to the largest face
    faces_result = []
    for img_orig, largest_face in zip(images, largest_faces):
        # Skip if no face is detected
        if largest_face is None:
            continue
        counter_has_face+=1
        (x,y,w,h) = largest_face
        # Calculate the coordinates for the square face crop
        crop_x1 = x 
        crop_y1 = y 
        crop_x2 = x + w
        crop_y2 = y + h
        # Check if the crop coordinates are within the image boundaries
        if crop_x1 < 0 or crop_y1 < 0 or crop_x2 > img_orig.shape[1] or crop_y2 > img_orig.shape[0]:
            # If the crop coordinates are outside the image boundaries, use a center crop instead
            min_dim = min(img_orig.shape[0], img_orig.shape[1])
            crop_x1 = (img_orig.shape[1] - min_dim) // 2
            crop_y1 = (img_orig.shape[0] - min_dim) // 2
            img_cropped = img_orig[crop_y1:crop_y1 + min_dim, crop_x1:crop_x1 + min_dim, :]
        else:
            # Crop the image based on the calculated coordinates
            img_cropped = img_orig[crop_y1:crop_y2, crop_x1:crop_x2, :]
        # Resize the cropped image to 512x512
        img = cv2.resize(img_cropped, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        faces_result.append(img)
        
    # Apply the mask to the original image
    mask_result = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    deeplab = make_deeplab(device)
    print("Applying mask...")
    for img_orig, largest_face in zip(images, largest_faces):
        # Skip if no face is detected
        if largest_face is None:
            continue
        # Apply the mask to the original image
        mask = apply_deeplab(deeplab, img_orig, device)
        masked_image = cv2.bitwise_and(img_orig,img_orig, mask=mask.astype(np.uint8))
        mask_result.append(masked_image)
    
    origin_images=[]
    for img_orig, largest_face in zip(images, largest_faces):
        # Skip if no face is detected
        if largest_face is None:
            continue
        origin_images.append(img_orig)
    
    # Create the output directory if it doesn't exist
    os.makedirs(OUT_DIR, exist_ok=True)

    # Save the image to the output directory
    for fileindex, img in enumerate(faces_result):
        os.makedirs(OUT_DIR+"/faceonly", exist_ok=True)
        output_path = os.path.join(OUT_DIR+"/faceonly", str(fileindex) + "_faceonly.jpg")
        cv2.imwrite(output_path, img)
    for fileindex, img in enumerate(mask_result):
        os.makedirs(OUT_DIR+"/masked", exist_ok=True)
        output_path = os.path.join(OUT_DIR+"/masked", str(fileindex) + "_masked.jpg")
        cv2.imwrite(output_path, img)
    for fileindex, img in enumerate(origin_images):
        os.makedirs(OUT_DIR+"/unmasked", exist_ok=True)
        output_path = os.path.join(OUT_DIR+"/unmasked", str(fileindex) + "_unmasked.jpg")
        cv2.imwrite(output_path, img)

    # Print the number of original images and the number of processed images
    num_original_images = len(os.listdir(IMAGE_DIR))
    print(f"Total number of original images: {num_original_images}")
    print(f"Total number of processed images that have faces: {counter_has_face}")

    return None

if __name__ == "__main__":
    main()