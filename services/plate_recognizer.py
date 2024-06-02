import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO

from consts import VEHICLE_CLASS_IDS
from services.car_info import CarInfo
from services.image_processing import ImageProcessing

coco_model = YOLO('models/yolov8n.pt')
carplate_model = YOLO('models/carplate_detection.pt')


class PlateRecognizer:
    @staticmethod
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

    @staticmethod
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

    def get_car_info(self, image: np.ndarray):
        vehicle_detections = self.detect_vehicles(image)
        license_plate_detections = self.detect_license_plates(image, vehicle_detections)
        texts = ImageProcessing.read_license_plates(image, license_plate_detections)
        car_info = CarInfo().get_car_info(texts[0][0])

        return car_info
