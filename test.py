import cv2
import time
import math
import numpy as np
import pyttsx3
import threading
import tkinter as tk
from tkinter import messagebox
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Initialize camera and modules
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(maxHands=1)
classifier = Classifier(
    "C:/Users/OneDrive/Desktop/Sign-Language-detection-main/converted_keras10/keras_model10.h5",
    "C:/Users/OneDrive/Desktop/Sign-Language-detection-main/converted_keras10/labels10.txt")

offset = 20
imgSize = 300

# Updated to only include A–Z
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Text-to-Speech
tts = pyttsx3.init()

# Tkinter GUI
root = tk.Tk()
root.title("Smart Assistant")
root.geometry("550x400")
root.configure(bg="white")

gesture_var = tk.StringVar()
sentence_var = tk.StringVar()

tk.Label(root, text="Current Letter:", font=("Arial", 14), bg="white").pack(pady=10)
gesture_label = tk.Label(root, textvariable=gesture_var, font=("Arial", 28), fg="blue", bg="white")
gesture_label.pack()

tk.Label(root, text="Formed Sentence:", font=("Arial", 14), bg="white").pack(pady=10)
sentence_label = tk.Label(root, textvariable=sentence_var, font=("Arial", 18), fg="green", wraplength=480, bg="white")
sentence_label.pack(pady=10)

# Sentence logic
sentence = ""
last_prediction = ""
last_time = time.time()
cooldown = 1.5  # seconds between gestures
inactivity_threshold = 5  # seconds
last_detected_time = time.time()
auto_spoken = False

# --- Button Functions ---
def clear_sentence():
    global sentence
    sentence = ""
    sentence_var.set(sentence)

def speak_sentence():
    if sentence:
        tts.say(sentence)
        tts.runAndWait()

def save_sentence():
    if sentence:
        with open("sentences.txt", "a") as file:
            file.write(sentence + "\n")
        messagebox.showinfo("Saved", "Sentence saved to sentences.txt")
    else:
        messagebox.showwarning("Empty", "No sentence to save.")

# --- Buttons ---
frame = tk.Frame(root, bg="white")
frame.pack(pady=10)

tk.Button(frame, text="Clear Sentence", command=clear_sentence, bg="red", fg="white", width=15).grid(row=0, column=0, padx=5)
tk.Button(frame, text="Speak Sentence", command=speak_sentence, bg="green", fg="white", width=15).grid(row=0, column=1, padx=5)
tk.Button(frame, text="Save to File", command=save_sentence, bg="blue", fg="white", width=15).grid(row=0, column=2, padx=5)

# --- Background thread ---
def process_video():
    global sentence, last_prediction, last_time, last_detected_time, auto_spoken

    while True:
        success, img = cap.read()
        if not success:
            continue

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            auto_spoken = False
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w
            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                label = labels[index]

                current_time = time.time()
                if label != last_prediction and (current_time - last_time) > cooldown:
                    sentence += label
                    last_prediction = label
                    last_time = current_time

                gesture_var.set(label)
                sentence_var.set(sentence)
                cv2.putText(imgOutput, label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 3)

                last_detected_time = current_time

            except:
                continue

        else:
            current_time = time.time()
            if sentence and not auto_spoken and (current_time - last_detected_time) > inactivity_threshold:
                tts.say(sentence)
                tts.runAndWait()
                auto_spoken = True
                sentence = ""
                sentence_var.set(sentence)

        # Show camera feed
        cv2.imshow("Camera Feed", imgOutput)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.quit()

# Start video thread
thread = threading.Thread(target=process_video)
thread.daemon = True
thread.start()

# Launch GUI
root.mainloop()
