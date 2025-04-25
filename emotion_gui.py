import cv2
from deepface import DeepFace
from tkinter import *
from PIL import Image, ImageTk
import threading
import time

# GUI setup
window = Tk()
window.title("Real-Time Emotion Detection")
window.geometry("800x600")
window.configure(bg="#222222")

# GUI components
button_frame = Frame(window, bg="#222222")
button_frame.pack(pady=10)

start_button = Button(button_frame, text="Start Detection", font=("Helvetica", 14), bg="#4CAF50", fg="white")
start_button.grid(row=0, column=0, padx=10)

stop_button = Button(button_frame, text="Stop Detection", font=("Helvetica", 14), bg="#f44336", fg="white")
stop_button.grid(row=0, column=1, padx=10)

video_label = Label(window, bg="#222222")
video_label.pack(pady=20)

emotion_label = Label(window, text="Click Start to Detect Emotion", font=("Helvetica", 20), fg="white", bg="#222222")
emotion_label.pack(pady=10)

# Global control flags
stop_camera = False
camera_running = False

def run_camera():
    global stop_camera, camera_running
    cap = cv2.VideoCapture(0)
    last_analysis_time = time.time()

    while not stop_camera:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze once per second
        if time.time() - last_analysis_time >= 1:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant_emotion = result[0]['dominant_emotion']
            except Exception:
                dominant_emotion = "No Face Detected"
            emotion_label.config(text=f"Emotion: {dominant_emotion}")
            last_analysis_time = time.time()

        # Display video
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        time.sleep(0.03)

    cap.release()
    camera_running = False
    emotion_label.config(text="Detection Stopped")

def start_camera():
    global stop_camera, camera_running
    if not camera_running:
        stop_camera = False
        camera_running = True
        emotion_label.config(text="Detecting Emotion...")
        threading.Thread(target=run_camera).start()

def stop_camera_func():
    global stop_camera
    stop_camera = True

def on_closing():
    stop_camera_func()
    window.destroy()

# Bind buttons
start_button.config(command=start_camera)
stop_button.config(command=stop_camera_func)

window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()
