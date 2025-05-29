import os
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2

# Set up the model and processor
processor = AutoImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")

# Specify the images directory
images_dir = "images"


def process_frame(frame):
    # Convert BGR (OpenCV format) to RGB (PIL format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    image = Image.fromarray(frame_rgb)

    # Prepare the frame for the model
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate the output to match the input frame size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(frame.shape[0], frame.shape[1]),
        mode="bicubic",
        align_corners=False,
    )

    # Normalize the output for visualization
    output = prediction.squeeze().cpu().numpy()
    output = (output - output.min()) / (output.max() - output.min()) * 255
    return output.astype('uint8')


def process_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # Prepare the image for the model
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate the output to match the input image size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Normalize the output for visualization
    output = prediction.squeeze().cpu().numpy()
    output = (output - output.min()) / (output.max() - output.min()) * 255
    depth_image = Image.fromarray(output.astype('uint8'))

    # Create output filename
    output_path = os.path.join("output", os.path.basename(image_path))
    os.makedirs("output", exist_ok=True)

    # Save the depth map
    depth_image.save(output_path)
    print(f"Processed {image_path} -> {output_path}")


def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output filename
    output_path = os.path.join("output", os.path.splitext(os.path.basename(video_path))[0] + "_depth.mp4")
    os.makedirs("output", exist_ok=True)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), False)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        depth_frame = process_frame(frame)
        out.write(depth_frame)

        # Update progress
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"\rProcessing video: {progress:.1f}% complete", end="")

    print(f"\nProcessed {video_path} -> {output_path}")

    # Release everything
    cap.release()
    out.release()


def process_all_files():
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} directory not found!")
        return

    for file in os.listdir(images_dir):
        file_path = os.path.join(images_dir, file)
        file_lower = file.lower()

        try:
            if file_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                process_image(file_path)
            elif file_lower.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                process_video(file_path)
            else:
                print(f"Skipping unsupported file: {file}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")


# Run the processing
if __name__ == "__main__":
    process_all_files()