import cv2
from fer import FER
import tkinter as tk
from PIL import Image, ImageTk
import sys

# Map FER emotions to emojis
EMOJI_MAP = {
    "happy": "😄",
    "angry": "😠",
    "surprise": "😲",
    "sad": "😢",
    "disgust": "🤢",
    "fear": "😨",
    "neutral": "😐"
}

def get_top_emotion(emotions_dict):
    """
    Finds the emotion with the highest score in the dictionary.
    """
    if not emotions_dict:
        return None
    # Filter out keys with a score of 0.0 to focus on positive detection
    active_emotions = {k: v for k, v in emotions_dict.items() if v > 0.0}
    if not active_emotions:
        return None
    return max(active_emotions, key=active_emotions.get)

class EmojiCameraApp:
    def _init_(self, root):
        """Initializes the Tkinter application, camera, and FER detector."""
        self.root = root
        self.root.title("Emoji Camera (Tkinter)")

        # --- UI Elements ---

        # Video frame label (to display webcam feed)
        self.video_label = tk.Label(root)
        self.video_label.pack(padx=10, pady=(10, 0))

        # Emoji label (big font for the detected emotion)
        # Using a reliable emoji font like 'Segoe UI Emoji' or just relying on system fonts.
        font_size = 64
        # Check if platform is macOS, where 'Apple Color Emoji' is often preferred
        if sys.platform == "darwin":
            emoji_font = ("Apple Color Emoji", font_size)
        else:
            emoji_font = ("Arial", font_size) # Fallback to a common font

        self.emoji_label = tk.Label(root, text="❓", font=emoji_font)
        self.emoji_label.pack(pady=10)

        # Text label for emotion description and score
        self.text_label = tk.Label(root, text="Detecting...", font=("Arial", 16))
        self.text_label.pack(pady=(0, 10))

        # --- Initialization ---

        # Initialize camera (0 is usually the default webcam)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.text_label.config(text="Error: Could not open camera. Check camera permissions/connection.")
            print("Error: Could not open camera.")
            self.detector = None # Prevent FER initialization if camera fails
            return

        # Set a slightly lower resolution for better performance, if possible
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Initialize FER detector (using mtcnn=False for faster, less accurate detection)
        try:
            # The 'fer' library may print warnings, which is normal.
            self.detector = FER(mtcnn=False)
        except Exception as e:
            self.text_label.config(text=f"Error initializing FER: {e}")
            self.detector = None
            print(f"Error initializing FER: {e}")
            return


        # Start updating frames
        self.update_frame()

        # Close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_frame(self):
        """Reads a frame, detects emotion, updates UI, and schedules the next frame."""
        # Only proceed if the camera and detector were successfully initialized
        if not self.cap or not self.cap.isOpened() or not self.detector:
            self.root.after(1000, self.on_close) # Attempt to close gracefully if initialization failed
            return

        ret, frame = self.cap.read()
        if not ret:
            self.text_label.config(text="Failed to grab frame.")
            # Schedule next frame even on failure to try again
            self.root.after(30, self.update_frame)
            return

        # Flip the frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)
        # Convert BGR (OpenCV format) to RGB (PIL format)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Emotion Detection ---
        results = self.detector.detect_emotions(rgb)

        current_emoji = "❓"
        emotion_text = "No face detected"

        if results:
            # The FER library returns a list of detections; we focus on the first detected face
            r = results[0]
            emotions = r["emotions"]
            top_emotion = get_top_emotion(emotions)
            (x, y, w, h) = r["box"] # Bounding box coordinates

            if top_emotion:
                current_emoji = EMOJI_MAP.get(top_emotion, "❓")
                score = emotions[top_emotion]
                # Format the text with the top emotion and its confidence score
                emotion_text = f"{top_emotion.capitalize()} ({score * 100:.1f}%)"

                # Optionally draw a bounding box around the detected face in the BGR frame
                # (Need to convert back to BGR for drawing, then back to RGB for display)
                bgr_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Put the emoji next to the box
                cv2.putText(bgr_frame, current_emoji, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB) # Convert back to RGB

            else:
                emotion_text = "Face detected, but low confidence"


        # --- UI Update ---

        # Update emojis and text labels
        self.emoji_label.config(text=current_emoji)
        self.text_label.config(text=emotion_text)

        # Convert the processed RGB frame (numpy array) to a Tkinter image (ImageTk.PhotoImage)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk # Keep a reference
        self.video_label.config(image=imgtk)

        # Schedule the next frame update (refresh rate of approx 30ms or ~33 FPS)
        self.root.after(30, self.update_frame)

    def on_close(self):
        """Releases the camera and destroys the Tkinter window."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("Camera released.")
        self.root.destroy()
        print("Application closed.")