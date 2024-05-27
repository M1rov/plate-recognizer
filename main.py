import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

from consts import VEHICLE_CLASS_IDS
from utils import read_license_plates

# Load the COCO model for vehicle detection
coco_model = YOLO('models/yolov8n.pt')
carplate_model = YOLO('models/carplate_detection.pt')


def detect_vehicles(image: np.ndarray) -> list[list[tuple]]:
    """
    Detects vehicles in an image using YOLO model.
    """

    # Detect vehicles
    detections = coco_model(image)[0]
    results = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in VEHICLE_CLASS_IDS:
            results.append([x1, y1, x2, y2, score])

    return results


def detect_license_plates(image: np.ndarray, vehicle_detections: list[list[tuple]]) -> list[list[tuple]]:
    """
    Detects license plates within vehicle bounding boxes.
    """
    license_plate_detections = []

    for vehicle in vehicle_detections:
        x1, y1, x2, y2, _ = vehicle
        vehicle_crop = image[int(y1):int(y2), int(x1):int(x2)]

        # Detect license plates in the vehicle crop
        plates = carplate_model(vehicle_crop)[0]
        for plate in plates.boxes.data.tolist():
            px1, py1, px2, py2, plate_score, plate_class_id = plate
            # Adjust coordinates to the original image
            plate_x1 = x1 + px1
            plate_y1 = y1 + py1
            plate_x2 = x1 + px2
            plate_y2 = y1 + py2
            license_plate_detections.append([plate_x1, plate_y1, plate_x2, plate_y2, plate_score])

    return license_plate_detections


def draw_detections(image: np.ndarray, vehicle_detections: list[list[tuple]], license_plate_detections: list[list[tuple]]):
    """
    Draws bounding boxes for detected vehicles and license plates on the image.
    """
    for vehicle in vehicle_detections:
        x1, y1, x2, y2, score = vehicle
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image, f"Vehicle {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    for plate in license_plate_detections:
        x1, y1, x2, y2, score = plate
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"Plate {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def main():
    image_path = 'image2.jpeg'
    image = cv2.imread(image_path)
    vehicle_detections = detect_vehicles(image)
    license_plate_detections = detect_license_plates(image, vehicle_detections)

    texts = read_license_plates(image, license_plate_detections)
    # draw_detections(image, vehicle_detections, license_plate_detections)
    print(texts)


if __name__ == "__main__":
    main()
