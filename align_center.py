
import os
import glob
import cv2
import numpy as np

SHOW = True            
MAX_SIZE = 900         
DEBUG_FOLDER = "debug" 
 
 
def _to_uint8_bgr(image):
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        img = (img - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(img)
        img = img.astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img
 
 
def _gui_available():
    if os.name == "nt" or os.uname().sysname == "Darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
 
 
def show_image(image, title="image", wait=True):
    if not SHOW:
        return
    img = _to_uint8_bgr(image)
    h, w = img.shape[:2]
    scale = min(1.0, MAX_SIZE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    if _gui_available():
        try:
            cv2.imshow(title, img)
            if wait:
                cv2.waitKey(0)
                cv2.destroyWindow(title)
            else:
                cv2.waitKey(1)
            return
        except cv2.error:   # opencv-python-headless
            pass
    os.makedirs(DEBUG_FOLDER, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in title)
    path = os.path.join(DEBUG_FOLDER, f"{safe}.png")
    cv2.imwrite(path, img)
    print(f"[show_image] окно недоступно, сохранено: {path}")
 
 
def show_compare(before, after, title="до | после", wait=True):
    """Показывает 'до' и 'после' рядом с подписями."""
    if not SHOW:
        return
    a, b = _to_uint8_bgr(before), _to_uint8_bgr(after)
    h = min(a.shape[0], b.shape[0])
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    for img, text in ((a, "before"), (b, "after")):
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 2, cv2.LINE_AA)
    show_image(np.hstack([a, b]), title, wait)
 

INPUT_FOLDER = "pictures"
OUTPUT_FOLDER = "aligned"


def find_sun_center(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    show_image(mask, "маска Солнца")
     
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea)

    (center_x, center_y), radius = cv2.minEnclosingCircle(biggest)
    return center_x, center_y


def move_sun_to_center(image, center_x, center_y, base_center_x, base_center_y):
    height, width = image.shape[:2]
    shift_x = base_center_x - center_x
    shift_y = base_center_y - center_y
    
    matrix = np.float32([[1, 0, shift_x],
                         [0, 1, shift_y]])
    result = cv2.warpAffine(image, matrix, (width, height))
    show_compare(image, result, "сдвиг")
    return result


os.makedirs(OUTPUT_FOLDER, exist_ok=True)
paths = glob.glob(os.path.join(INPUT_FOLDER, "*.png"))

base_image = cv2.imread(paths[0])
base_center_x, base_center_y = find_sun_center(base_image)
print(f"base:  x={base_center_x:.2f}, y={base_center_y:.2f}")

for path in paths[1:]:
    image = cv2.imread(path)

    center_x, center_y = find_sun_center(image)
    result = move_sun_to_center(image, center_x, center_y, base_center_x, base_center_y)

    #Check the new center after moving
    new_x, new_y = find_sun_center(result)
    print(f"было ({center_x:.2f}, {center_y:.2f})  ->  стало ({new_x:.2f}, {new_y:.2f})")



    file_name = os.path.basename(path)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, file_name), result)
    print(file_name, "готово")
