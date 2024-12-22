import cv2

# Load a pre-trained model for person detection (e.g., Haar Cascade or YOLO)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

def monitor_crowd():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam; replace with a video path for testing
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = detector.detectMultiScale(gray, 1.1, 3)

        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Detected: {len(bodies)} people", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Crowd Tracker", frame)

        if cv2.waitKey(1) == 27:  # Exit on 'ESC'
            break

    cap.release()
    cv2.destroyAllWindows()

monitor_crowd()
