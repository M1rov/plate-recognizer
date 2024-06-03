import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pytesseract import pytesseract

from services.plate_validation import PlateValidation


class ImageProcessing:
    @staticmethod
    def show_image(image: np.ndarray):
        # Display the image
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    @staticmethod
    def process_image(image: np.ndarray):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise and improve binarization
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Binarize the image using adaptive thresholding
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Perform morphological operations to remove small noise and enhance text
        kernel = np.ones((3, 3), np.uint8)
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        return morphed

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
    def read_license_plate(image: np.ndarray, license_plate_detection: list[float]) -> tuple:
        x1, y1, x2, y2, score = license_plate_detection
        license_plate_crop = image[int(y1):int(y2), int(x1): int(x2), :]

        deskewed_license_plate = ImageProcessing.deskew(license_plate_crop)
        license_plate_processed = ImageProcessing.process_image(deskewed_license_plate)
        ImageProcessing.show_image(license_plate_processed)

        license_plate_text = pytesseract.image_to_string(license_plate_processed, config=r'--oem 3 --psm 8')

        print(license_plate_text, 'lp text')
        text = PlateValidation.format_license_plate(license_plate_text.upper())
        print(text, 'here')

        if PlateValidation.license_complies_format(text):
            return text, score
        else:
            raise Exception(f'Не вдалось розпізнати номера: {text}')