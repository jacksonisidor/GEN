from pathlib import Path
import sys
import cv2
import os
import csv
import time

# ------------------ Paths ------------------

script_dir = Path(__file__).parent
input_dir = script_dir / "output_images"
output_dir = script_dir / "altered_images"
output_dir.mkdir(exist_ok=True)

IMAGE_NAME = sys.argv[1]
IMAGE_PATH = input_dir / (IMAGE_NAME + ".jpg")
output_path = output_dir / f"boxed_{IMAGE_NAME}.jpg"

print("Will save to:", output_path)

csv_path = script_dir / "annotations.csv"

print("Trying to load:", IMAGE_PATH)
dictionary = {"lp", "eo", "logo", "ss"}
img = cv2.imread(str(IMAGE_PATH))
if img is None:
    raise FileNotFoundError(f"Could not load image at path: {IMAGE_PATH}")

# Image that accumulates all permanent boxes
img_with_boxes = img.copy()

# ------------------ Globals ------------------

drawing = False
ix, iy = -1, -1
boxes = []  # list of (x1, y1, x2, y2, label)

# ------------------ CSV Setup ------------------

# Create CSV with header if it doesn't exist
if not csv_path.exists():
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x1", "y1", "x2", "y2", "label"])

# ------------------ Mouse Callback ------------------

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img_with_boxes, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Show preview: permanent boxes + current temp box
            temp_img = img_with_boxes.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1, x2, y2 = ix, iy, x, y

        # Normalize coordinates (drag in any direction)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Draw permanently on the image
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Image", img_with_boxes)
        cv2.waitKey(1)  # force OpenCV to render the box immediately
        while True:
            label = input(f"Enter label for box ({x1},{y1}) -> ({x2},{y2}) [type 'redo' to redraw]: ").strip()
            if label.lower() == "redo":
                # Remove temporary box
                img_with_boxes = img.copy()
                # Re-draw all previous committed boxes
                for bx, by, bx2, by2, blabel in boxes:
                    cv2.rectangle(img_with_boxes, (bx, by), (bx2, by2), (0, 255, 0), 2)
                print("Redoing box, draw again...")
                return  # exit callback, user will redraw
            elif label not in dictionary:
                print("Invalid label")
            else:
                break  # valid label entered

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([IMAGE_NAME, x1, y1, x2, y2, label])

        print(f"Added box with label '{label}'")



# ------------------ UI Loop ------------------

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_rectangle)

print("Instructions:")
print("- Click and drag to draw boxes")
print("- After each box, type a label in the terminal and press Enter")
print("- Boxes stay on the image")
print("- Press 'q' to quit")

while True:
    cv2.imshow("Image", img_with_boxes)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        success = cv2.imwrite(str(output_path), img_with_boxes)
        print("Save success:", success)
        print("Saved to:", output_path)
        break

cv2.destroyAllWindows()
