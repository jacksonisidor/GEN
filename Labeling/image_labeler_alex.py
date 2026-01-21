from pathlib import Path
import sys
import cv2
import csv


# ------------------ File Paths ------------------
script_dir = Path(__file__).resolve().parent
input_dir = script_dir / "standardized_training_images"
output_dir = script_dir / "altered_images_alex"
output_dir.mkdir(exist_ok=True)

IMAGE_NAME = sys.argv[1]
IMAGE_PATH = input_dir / f"{IMAGE_NAME}.jpg"
output_path = output_dir / f"boxed_{IMAGE_NAME}.jpg"
csv_path = script_dir / "annotations_alex.csv"

objects = {"lp", "eo", "logo", "ss"}

img = cv2.imread(str(IMAGE_PATH))
if img is None:
    raise FileNotFoundError(f"Could not load image at path: {IMAGE_PATH}")

# -------- Speed / Resolution --------

DISPLAY_SCALE = 0.5

if DISPLAY_SCALE != 1.0:
    disp_img = cv2.resize(img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
else:
    disp_img = img

box_thickness = 2

# -------- Drawing --------
    
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
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness, lineType=cv2.LINE_8)
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
        cv2.rectangle(base, (x1, y1), (x2, y2), (0, 255, 0), box_thickness, lineType=cv2.LINE_8)

        # Ask label
        while True:

            label = input(
                f"Enter label [{', '.join(sorted(objects))}] or 'redo' or 'clean': "
            ).strip().lower()

            if label == "redo":
                # remove the last drawn rect by rebuilding base
                base = disp_img.copy()
                for bx, by, bx2, by2, bl in boxes:
                    cv2.rectangle(base, (bx, by), (bx2, by2), (0, 255, 0), box_thickness, lineType=cv2.LINE_8)
                render()
                return
            
            if label == "clean":
                # wipe all boxes for this image (memory + CSV) and cancel current box
                remove_all_csv_rows_for_image(csv_path, IMAGE_NAME)
                clear_all_boxes()
                return

            if label in objects:
                break
            print("Invalid label.")

        boxes.append((x1, y1, x2, y2, label))

        # Save coords in ORIGINAL image coords
        ox1, oy1 = to_original_coords(x1, y1)
        ox2, oy2 = to_original_coords(x2, y2)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([IMAGE_NAME, ox1, oy1, ox2, oy2, label])

        render()

def remove_all_csv_rows_for_image(csv_path, image_name):
    """Remove ALL rows in the CSV corresponding to image_name. Rewrites the file."""
    if not csv_path.exists():
        return

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) <= 1:
        return

    header = rows[0]
    data = rows[1:]

    data = [r for r in data if len(r) > 0 and r[0] != image_name]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)


# UI
        
def clear_all_boxes():
    global boxes, base
    boxes = []
    base = disp_img.copy()
    render()

        
def to_display_coords(x, y):
    if DISPLAY_SCALE == 1.0:
        return x, y
    return int(x * DISPLAY_SCALE), int(y * DISPLAY_SCALE)

def load_existing_boxes():
    """Load existing annotations for this image from CSV and draw them on base."""
    global boxes, base

    if not csv_path.exists():
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["image"] != IMAGE_NAME:
                continue

            # stored in ORIGINAL coords
            ox1 = int(float(row["x1"]))
            oy1 = int(float(row["y1"]))
            ox2 = int(float(row["x2"]))
            oy2 = int(float(row["y2"]))
            label = row["label"].strip().lower()

            # convert to DISPLAY coords for drawing + in-memory boxes
            x1, y1 = to_display_coords(ox1, oy1)
            x2, y2 = to_display_coords(ox2, oy2)
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            boxes.append((x1, y1, x2, y2, label))

    # redraw base with all loaded boxes
    base = disp_img.copy()
    for bx, by, bx2, by2, bl in boxes:
        cv2.rectangle(base, (bx, by), (bx2, by2), (0, 255, 0), box_thickness, lineType=cv2.LINE_8)


cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image", on_mouse)

load_existing_boxes()
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
            cv2.rectangle(out, (ox1, oy1), (ox2, oy2), (0, 255, 0), box_thickness, lineType=cv2.LINE_8)

        success = cv2.imwrite(str(output_path), out)
        print("Save success:", success)
        print("Saved to:", output_path)
        break

cv2.destroyAllWindows()