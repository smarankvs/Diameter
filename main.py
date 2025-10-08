import cv2
import numpy as np

REFERENCE_SIZE_MM = 22

def get_largest_contour(contours, exclude=None):
    max_area = 0
    best_cnt = None
    for cnt in contours:
        if exclude is not None and np.array_equal(cnt, exclude):
            continue
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
    return best_cnt

def skeletonize(img):
    """Returns skeletonized binary image for centerline extraction."""
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False

    img = img.copy()
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            done = True
    return skel

def measure_diameter_along_centerline(thread_mask, px_to_mm, img, num_samples=15):
    skeleton = skeletonize(thread_mask)
    points = np.column_stack(np.where(skeleton > 0))

    # Sample points along skeleton, evenly spaced
    if len(points) < 2:
        return None

    idxs = np.linspace(0, len(points) - 1, num_samples, dtype=int)
    diameters_px = []
    for idx in idxs:
        y, x = points[idx]
        # To find diameter at (x,y), scan in perpendicular directions
        # Estimate tangent from neighbors
        neighbors = points[max(0, idx-2):min(len(points), idx+3)]
        if len(neighbors) >= 2:
            dy = neighbors[-1][0] - neighbors[0][0]
            dx = neighbors[-1][1] - neighbors[0][1]
            norm = np.sqrt(dx**2+dy**2)
            if norm > 0:
                vx, vy = -dy/norm, dx/norm  # Perpendicular vector
            else:
                vx, vy = 0, 1
        else:
            vx, vy = 0, 1

        # Scan outwards from (x, y) until leaving mask on both sides
        for dist in range(1, max(thread_mask.shape)//3):
            x1 = int(x + vx * dist)
            y1 = int(y + vy * dist)
            x2 = int(x - vx * dist)
            y2 = int(y - vy * dist)
            if (0 <= y1 < thread_mask.shape[0] and 0 <= x1 < thread_mask.shape[1] and 
                0 <= y2 < thread_mask.shape[0] and 0 <= x2 < thread_mask.shape[1]):
                if thread_mask[y1, x1] == 0 or thread_mask[y2, x2] == 0:
                    diam_px = dist * 2
                    diameters_px.append(diam_px)
                    # Draw measurement line for visualization
                    cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 1)
                    break

    if len(diameters_px) > 0:
        diameters_mm = np.array(diameters_px) * px_to_mm
        mean_diam = np.mean(diameters_mm)
        std_diam = np.std(diameters_mm)
        return mean_diam, std_diam, img
    else:
        return None

def measure_thread(image_path, output_path="annotated_output.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid path.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find circular reference
    ref_contour = None
    ref_size_px = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        ar = w / h if h != 0 else 0
        if area > 100 and 0.8 < ar < 1.2:  # Circular-ish
            ref_contour = cnt
            ref_size_px = max(w, h)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Reference", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            break

    # Find largest contour as thread/object
    thread_contour = get_largest_contour(contours, exclude=ref_contour)

    diameters_mm = None
    if ref_size_px and thread_contour is not None:
        px_to_mm = REFERENCE_SIZE_MM / ref_size_px

        # Make mask of thread
        thread_mask = np.zeros_like(gray)
        cv2.drawContours(thread_mask, [thread_contour], -1, 255, thickness=-1)
        result = measure_diameter_along_centerline(thread_mask, px_to_mm, img)
        if result is None:
            measured_mm = None
        else:
            mean_diam, std_diam, img = result
            measured_mm = round(mean_diam, 2)
            cv2.putText(img, f"Diameter: {measured_mm} mm", 
                        (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
    else:
        measured_mm = None

    img = cv2.resize(img, (800, 600))
    cv2.imshow("Measured Threads", img)
    cv2.waitKey(0)
    cv2.imwrite(output_path, img)
    print(f"Measured Diameter: {measured_mm} mm" if measured_mm else "Measurement failed.")
    return measured_mm

# Example usage
measure_thread(r"D:\Smaran_Required\ASPL\Diameter\bottle1.png")

#error is arond 0.5 cm
