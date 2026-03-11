# import asyncio
# from asyncio import WindowsSelectorEventLoopPolicy
import asyncio, sys
from flask import Flask, render_template, request, jsonify
import g4f
from deepface import DeepFace
import cv2
import numpy as np
from collections import Counter
import threading
import time


if sys.platform == "win32":
    from asyncio import WindowsSelectorEventLoopPolicy
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
# Set the event loop policy for Windows (if applicable)
# asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

app = Flask(__name__)

# Global variable to store captured emotions
emotion_data = []

def capture_emotions():
    """Capture emotions from webcam for 10 seconds and return the dominant emotion."""
    global emotion_data
    emotion_data = []  # Reset emotion data
    cap = cv2.VideoCapture(0)  # Open the default webcam
    if not cap.isOpened():
        return "Error: Could not access webcam."

    start_time = time.time()
    while time.time() - start_time < 10:  # Run for 10 seconds
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Analyze emotions using DeepFace
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']  # Extract dominant emotion
            emotion_data.append(dominant_emotion)
        except Exception as e:
            print(f"Emotion detection error: {e}")
            continue

        # Display the frame with emotion label (for user feedback)
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detection - Press Q to Stop', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
            break

    cap.release()
    cv2.destroyAllWindows()

    # Determine the most frequent emotion
    if emotion_data:
        emotion_counts = Counter(emotion_data)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        return f"Detected dominant emotion: {dominant_emotion}"
    return "No emotions detected."

def generate_response(user_input):
    """
    Generate a response using GPT-4 with a system prompt to ensure the chatbot
    discusses mental health topics and common health concerns while suggesting
    general over-the-counter tablets when relevant.
    """
    try:
        if user_input.lower().strip() == "restart":
            return "Chat has been restarted. How can I assist you today?"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Health and Mental Wellness Support Chatbot designed to assist users with mental health concerns, "
                    "common health issues, and general well-being. "
                    "You provide empathetic support, coping strategies, and general health suggestions for mild conditions like fever, headache, stress, insomnia, acidity, and colds. "
                    "When relevant, you may suggest specific over-the-counter (OTC) tablets along with a disclaimer. Here are some examples:\n\n"
                    "- **Fever:** Paracetamol (e.g., Calpol, Crocin, Tylenol)\n"
                    "- **Headache:** Ibuprofen (e.g., Advil, Brufen) or Paracetamol (e.g., Crocin, Tylenol)\n"
                    "- **Cold & Cough:** Antihistamines (e.g., Cetirizine, Allegra) or Cough Syrup (e.g., Benadryl, Dextromethorphan)\n"
                    "- **Stress & Anxiety:** Magnesium supplements (e.g., Magnesiocard) or Herbal options (e.g., Ashwagandha)\n"
                    "- **Sleep Issues:** Melatonin (e.g., Natrol Melatonin, Circadin)\n"
                    "- **Acidity:** Antacids (e.g., Gelusil, Rantac, Omeprazole)\n\n"
                    "You **must always include the disclaimer:** 'I am not a doctor; please consult a healthcare professional before taking any medication.' "
                    "If the input mentions a detected emotion (e.g., 'Detected dominant emotion: sad'), respond empathetically and offer support based on that emotion. "
                    "Do not provide medical diagnoses or prescription medications. If a user asks about anything unrelated to health, politely inform them that you specialize in health and wellness support."
                )
            },
            {"role": "user", "content": user_input}
        ]
        response = g4f.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.6,
            top_p=0.9
        )
        return response.strip() if response else "Sorry, I couldn't process your query."
    except Exception as e:
        return f"Error: {e}"

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/chatbot")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    response = generate_response(user_input)
    return jsonify({"response": response})

@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    # Run emotion capture in a separate thread to avoid blocking
    thread = threading.Thread(target=capture_emotions)
    thread.start()
    thread.join()  # Wait for the thread to complete
    emotion_result = capture_emotions()  # Get the result
    response = generate_response(emotion_result)
    return jsonify({"emotion": emotion_result, "response": response})

if __name__ == "__main__":
    app.run(debug=True)