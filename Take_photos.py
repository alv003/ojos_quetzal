#Create a cv2 window to take 25 photos and give 2 seconds to change the camera position

import cv2
import time

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create a window named "Take Photos"
cv2.namedWindow("Take Photos")

# Take 25 photos
for i in range(25):
    # Give 2 seconds to change the camera position
    time.sleep(2)

    # Capture the frame
    ret, frame = cap.read()

    # Check if the frame is captured correctly
    if not ret:
        print("Error: Could not capture frame.")
        break

    # Show the frame in the window
    cv2.imshow("Take Photos", frame)

    # Save the frame as a .jpg file
    cv2.imwrite(f"photo_{i}.jpg", frame)

    # Wait for 1 second
    cv2.waitKey(1000)

# Release the VideoCapture object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()

# Print a message

print("Photos taken successfully.")
