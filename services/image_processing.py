import math

import cv2
import easyocr as easyocr
import numpy as np

from services.plate_validation import PlateValidation

reader = easyocr.Reader(['en'])


class ImageProcessing:
    @staticmethod
    def process_image(image: np.ndarray):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gray

    @staticmethod
    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    @staticmethod
    def compute_skew(src_img):

        if len(src_img.shape) == 3:
            h, w, _ = src_img.shape
        elif len(src_img.shape) == 2:
            h, w = src_img.shape
        else:
            print('upsupported image type')

        img = cv2.medianBlur(src_img, 3)

        edges = cv2.Canny(img, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30, minLineLength=w / 4.0, maxLineGap=h / 4.0)
        angle = 0.0
        nlines = lines.size

        # print(nlines)
        cnt = 0
        for x1, y1, x2, y2 in lines[0]:
            ang = np.arctan2(y2 - y1, x2 - x1)
            # print(ang)
            if math.fabs(ang) <= 30:  # excluding extreme rotations
                angle += ang
                cnt += 1

        if cnt == 0:
            return 0.0
        return (angle / cnt) * 180 / math.pi

    @staticmethod
    def deskew(src_img):
        return ImageProcessing.rotate_image(src_img, ImageProcessing.compute_skew(src_img))

    @staticmethod
    def get_centered_object(image: np.ndarray, detections: list[list[float]]) -> list[float]:
        img_height, img_width = image.shape[:2]
        center_x, center_y = img_width / 2, img_height / 2

        min_distance = float('inf')
        centered_object = None

        for detection in detections:
            x_min, y_min, x_max, y_max = detection[:4]
            object_center_x = (x_min + x_max) / 2
            object_center_y = (y_min + y_max) / 2

            distance = np.sqrt((center_x - object_center_x) ** 2 + (center_y - object_center_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                centered_object = detection

        return centered_object

    @staticmethod
    def read_license_plate(image: np.ndarray, license_plate_detection: list[float]) -> tuple:
        x1, y1, x2, y2, score = license_plate_detection
        license_plate_crop = image[int(y1):int(y2), int(x1): int(x2), :]

        deskewed_license_plate = ImageProcessing.deskew(license_plate_crop)
        license_plate_processed = ImageProcessing.process_image(deskewed_license_plate)

        result = reader.readtext(license_plate_processed, detail=0)
        if not len(result):
            raise Exception('Не вдалось прочитати номерні знаки, спробуйте інше зображення!')
        license_plate_text = result[0]

        print(license_plate_text, 'lp text')
        text = PlateValidation.format_license_plate(license_plate_text.upper())
        print(text, 'here')

        if PlateValidation.license_complies_format(text):
            return text, score
        else:
            raise Exception(f'Не вдалось розпізнати номера: {text}')