import cv2
import numpy as np

REFERENCE_SIZE_MM = 22  

def measure_thread(image_path, output_path="annotated_output.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid path.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blur, 50, 150)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ref_contour = None
    ref_size_px = None
    thread_contour = None
    thread_size_px = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        ar = w / h if h != 0 else 0
        if area > 100 and 0.8 < ar < 1.2:
            ref_contour = cnt
            ref_size_px = max(w, h)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Reference", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            break 


    max_area = 0
    for cnt in contours:
        if np.array_equal(cnt, ref_contour):
            continue  
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            thread_contour = cnt

    diameters_mm = None
    thread_diameter_px = None
    if ref_size_px and thread_contour is not None:
        x, y, w, h = cv2.boundingRect(thread_contour)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        thread_diameter_px = min(w, h)
        px_to_mm = REFERENCE_SIZE_MM / ref_size_px
        measured_mm = round(thread_diameter_px * px_to_mm, 2)
        diameters_mm = measured_mm
        label = f"Measured Diameter: {measured_mm} mm"
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

    img = cv2.resize(img, (800, 600))
    cv2.imshow("Measured Threads", img)
    cv2.waitKey(0)
    cv2.imwrite(output_path, img)

    return thread_diameter_px, diameters_mm


measure_thread(r"D:/Smaran_Required/ASPL/rope2.jpeg")