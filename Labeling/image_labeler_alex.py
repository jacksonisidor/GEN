from pathlib import Path
import sys
import cv2
import csv


# ------------------ Paths ------------------
script_dir = Path(__file__).resolve().parent
input_dir = script_dir / "standardized_training_images"
output_dir = script_dir / "altered_images_alex"
output_dir.mkdir(exist_ok=True)

IMAGE_NAME = sys.argv[1]
IMAGE_PATH = input_dir / f"{IMAGE_NAME}.jpg"
output_path = output_dir / f"boxed_{IMAGE_NAME}.jpg"
csv_path = script_dir / "annotations_alex.csv"

dictionary = {"lp", "eo", "logo", "ss"}

img = cv2.imread(str(IMAGE_PATH))
if img is None:
    raise FileNotFoundError(f"Could not load image at path: {IMAGE_PATH}")

# -------- BIG SPEED LEVER --------
# Change this to 0.5 or 0.33 if images are large (recommended on Mac)
DISPLAY_SCALE = 0.5

if DISPLAY_SCALE != 1.0:
    disp_img = cv2.resize(img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
else:
    disp_img = img

# permanent layer in DISPLAY coords
base = disp_img.copy()

drawing = False
ix, iy = -1, -1
boxes = []         # (x1,y1,x2,y2,label) in DISPLAY coords
last_render = base  # last image shown

def to_original_coords(x, y):
    if DISPLAY_SCALE == 1.0:
        return x, y
    return int(x / DISPLAY_SCALE), int(y / DISPLAY_SCALE)

# CSV header if missing
if not csv_path.exists():
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["image", "x1", "y1", "x2", "y2", "label"])

def render(temp_rect=None):
    """Render only when needed."""
    global last_render
    # Avoid copying unless we must draw a temp rect
    if temp_rect is None:
        last_render = base
    else:
        # copy only while dragging (still much less work than 60fps loop)
        frame = base.copy()
        x1, y1, x2, y2 = temp_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_8)
        last_render = frame

    cv2.imshow("Image", last_render)

def on_mouse(event, x, y, flags, param):
    global drawing, ix, iy, base, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        render((ix, iy, x, y))

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Update preview ONLY on mouse move
        render((ix, iy, x, y))

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        x1, y1, x2, y2 = ix, iy, x, y
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        # Draw permanent on base
        cv2.rectangle(base, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_8)

        # Ask label
        while True:
            label = input(
                f"Label for ({x1},{y1}) -> ({x2},{y2}) "
                f"[{', '.join(sorted(dictionary))}] or 'redo': "
            ).strip().lower()

            if label == "redo":
                # remove the last drawn rect by rebuilding base
                base = disp_img.copy()
                for bx, by, bx2, by2, bl in boxes:
                    cv2.rectangle(base, (bx, by), (bx2, by2), (0, 255, 0), 2, lineType=cv2.LINE_8)
                render()
                return

            if label in dictionary:
                break
            print("Invalid label.")

        boxes.append((x1, y1, x2, y2, label))

        # Save coords in ORIGINAL image coords
        ox1, oy1 = to_original_coords(x1, y1)
        ox2, oy2 = to_original_coords(x2, y2)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([IMAGE_NAME, ox1, oy1, ox2, oy2, label])

        render()

# UI
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image", on_mouse)
render()

print("Press 'q' to quit and save.")

while True:
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        # Save boxed image at ORIGINAL resolution
        out = img.copy()
        for x1, y1, x2, y2, label in boxes:
            ox1, oy1 = to_original_coords(x1, y1)
            ox2, oy2 = to_original_coords(x2, y2)
            cv2.rectangle(out, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2, lineType=cv2.LINE_8)

        success = cv2.imwrite(str(output_path), out)
        print("Save success:", success)
        print("Saved to:", output_path)
        break

cv2.destroyAllWindows()