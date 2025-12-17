import cv2
from fer import FER
import tkinter as tk
from PIL import Image, ImageTk
import sys

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

    if not emotions_dict:
        return None
    # Filter out keys with a score of 0.0 to focus on positive detection
    active_emotions = {k: v for k, v in emotions_dict.items() if v > 0.0}
    if not active_emotions:
        return None
    return max(active_emotions, key=active_emotions.get)

class EmojiCameraApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Emoji Camera (Tkinter)")

        
        self.video_label = tk.Label(root)
        self.video_label.pack(padx=10, pady=(10, 0))

       
        font_size = 64
       
        if sys.platform == "darwin":
            emoji_font = ("Apple Color Emoji", font_size)
        else:
            emoji_font = ("Arial", font_size) # Fallback to a common font

        self.emoji_label = tk.Label(root, text="❓", font=emoji_font)
        self.emoji_label.pack(pady=10)

       
        self.text_label = tk.Label(root, text="Detecting...", font=("Arial", 16))
        self.text_label.pack(pady=(0, 10))

       
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.text_label.config(text="Error: Could not open camera. Check camera permissions/connection.")
            print("Error: Could not open camera.")
            self.detector = None 
            return

        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

       
        try:
            
            self.detector = FER(mtcnn=False)
        except Exception as e:
            self.text_label.config(text=f"Error initializing FER: {e}")
            self.detector = None
            print(f"Error initializing FER: {e}")
            return


        self.update_frame()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_frame(self):
        
        if not self.cap or not self.cap.isOpened() or not self.detector:
            self.root.after(1000, self.on_close) 
            return

        ret, frame = self.cap.read()
        if not ret:
            self.text_label.config(text="Failed to grab frame.")
       
            self.root.after(30, self.update_frame)
            return

     
        frame = cv2.flip(frame, 1)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   
        results = self.detector.detect_emotions(rgb)

        current_emoji = "❓"
        emotion_text = "No face detected"

        if results:
           
            r = results[0]
            emotions = r["emotions"]
            top_emotion = get_top_emotion(emotions)
            (x, y, w, h) = r["box"] 

            if top_emotion:
                current_emoji = EMOJI_MAP.get(top_emotion, "❓")
                score = emotions[top_emotion]
               
                emotion_text = f"{top_emotion.capitalize()} ({score * 100:.1f}%)"

               
                bgr_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
                cv2.putText(bgr_frame, current_emoji, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB) # Convert back to RGB

            else:
                emotion_text = "Face detected, but low confidence"



        self.emoji_label.config(text=current_emoji)
        self.text_label.config(text=emotion_text)

        
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk # Keep a reference
        self.video_label.config(image=imgtk)

        
        self.root.after(30, self.update_frame)

    def on_close(self):
        """Releases the camera and destroys the Tkinter window."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("Camera released.")
        self.root.destroy()

        print("Application closed.")
