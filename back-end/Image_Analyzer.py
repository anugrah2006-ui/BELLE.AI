# Image_Analyzer.py
# BELLE.AI Aya Vision Analyzer (Cohere SDK)

import os
import base64
import mimetypes
import cv2
import cohere
from dotenv import load_dotenv
from tkinter import Tk, filedialog

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize Cohere Client
# Using c4ai-aya-vision-8b for BELLE Aya Vision
client = cohere.ClientV2(api_key=COHERE_API_KEY)

# -----------------------------
# CAMERA CAPTURE
# -----------------------------
def capture_from_camera(save_path="captured.jpg"):
    cap = cv2.VideoCapture(0)

    print("Camera started. Press SPACE to capture.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        cv2.imshow("BELLE Camera", frame)

        # SPACE key
        if cv2.waitKey(1) & 0xFF == 32:
            cv2.imwrite(save_path, frame)
            print(f"Saved image: {save_path}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return save_path


# -----------------------------
# UPLOAD FROM PC
# -----------------------------
def upload_from_pc():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        raise Exception("No file selected.")

    print(f"Selected file: {file_path}")
    return file_path


# -----------------------------
# CONVERT IMAGE TO BASE64
# -----------------------------
def encode_image(image_path):
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    return f"data:{mime};base64,{encoded}"


# -----------------------------
# AYA VISION ANALYZER
# -----------------------------
def analyze_with_aya(prompt, image_path):
    print("Sending image to Aya Vision via Cohere...")

    data_url = encode_image(image_path)

    try:
        response = client.chat(
            model="c4ai-aya-vision-8b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]
        )
        return response.message.content[0].text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


# -----------------------------
# MAIN ANALYZER
# -----------------------------
def analyze_image(prompt, source="upload"):
    """
    BELLE.AI Aya Vision Analyzer
    """

    if source == "cam":
        image_path = capture_from_camera()
    else:
        image_path = upload_from_pc()

    return analyze_with_aya(prompt, image_path)


# -----------------------------
# RUN PROGRAM
# -----------------------------
if __name__ == "__main__":
    print("--- BELLE.AI Aya Vision Analyzer ---")
    mode = input("Type 'cam' or 'upload': ").strip().lower()
    user_prompt = input("Enter your prompt: ").strip()

    if not user_prompt:
        user_prompt = "Describe this image in detail."

    try:
        result = analyze_image(user_prompt, source=mode)
        print("\nBELLE.AI says:\n", result)
    except Exception as e:
        print(f"An error occurred: {e}")
