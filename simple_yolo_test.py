import torch
import cv2

# Load YOLOv5 model (custom trained)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='lego_yolov5s.pt', source='local')  # Replace path

# Set model to evaluation mode
model.eval()

# Open webcam (0 = default cam)
cap = cv2.VideoCapture(0)

# Set video size (optional)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Render results on the frame
    annotated_frame = results.render()[0]

    # Display frame
    cv2.imshow('YOLOv5 LEGO Detection', annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
