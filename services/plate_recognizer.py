import time

import numpy as np
from ultralytics import YOLO

from consts import VEHICLE_CLASS_IDS
from services.car_info import CarInfo
from services.image_processing import ImageProcessing

coco_model = YOLO('models/yolov8n.pt')
carplate_model = YOLO('models/carplate_detection.pt')


class PlateRecognizer:
    @staticmethod
    def detect_vehicles(image: np.ndarray) -> list[list[float]]:
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
    def detect_license_plate(image: np.ndarray, vehicle_detection: list[float]) -> list[float]:
        """
        Detects license plates within vehicle bounding boxes.
        """

        x1, y1, x2, y2, _ = vehicle_detection
        vehicle_crop = image[int(y1):int(y2), int(x1):int(x2)]

        # Detect license plates in the vehicle crop
        plates = carplate_model(vehicle_crop)[0]

        if not len(plates):
            raise Exception('На цьому фото не видно номерних знаків, спробуйте завантажити інше!')

        centered_plate = ImageProcessing.get_centered_object(image, plates.boxes.data.tolist())

        px1, py1, px2, py2, plate_score, plate_class_id = centered_plate
        # Adjust coordinates to the original image
        plate_x1 = x1 + px1
        plate_y1 = y1 + py1
        plate_x2 = x1 + px2
        plate_y2 = y1 + py2
        license_plate_detection = [plate_x1, plate_y1, plate_x2, plate_y2, plate_score]

        return license_plate_detection

    def get_car_info(self, image: np.ndarray):
        start = time.perf_counter()
        vehicle_detections = self.detect_vehicles(image)
        if not len(vehicle_detections):
            raise Exception('На цьому фото не було знайдено автомобілів, спробуйте завантажити інше.')
        centered_vehicle = ImageProcessing.get_centered_object(image, vehicle_detections)
        license_plate_detection = self.detect_license_plate(image, centered_vehicle)
        text = ImageProcessing.read_license_plate(image, license_plate_detection)
        car_info = CarInfo().get_car_info(text[0])
        request_time = time.perf_counter() - start
        print(f'Фотографія була оброблена за {request_time}')
        return car_info
