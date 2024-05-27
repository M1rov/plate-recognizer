import string
import re

import cv2
import easyocr as easyocr
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def read_license_plates(image: np.ndarray, license_plate_detections: list[list[tuple]]) -> list[tuple[str, int]]:
    license_plate_texts = []
    for licence_plate in license_plate_detections:
        x1, y1, x2, y2, score = licence_plate
        license_plate_crop = image[int(y1):int(y2), int(x1): int(x2), :]

        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

        detections = reader.readtext(license_plate_crop_gray)
        for detection in detections:
            bbox, text, score = detection

            text = re.sub(r'[^A-Z0-9]', '', text.upper())

            if license_complies_format(text):
                license_plate_texts.append((format_license_plate(text), score))
            else:
                print('WRONG NUMBERS -', text)

    return license_plate_texts


def format_license_plate(text: str) -> str:
    license_plate = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_char_to_int, 3: dict_char_to_int, 4: dict_char_to_int,
               5: dict_char_to_int, 6: dict_int_to_char, 7: dict_int_to_char}
    for j in [0, 1, 2, 3, 4, 5, 6, 7]:
        if text[j] in mapping[j].keys():
            license_plate += mapping[j][text[j]]
        else:
            license_plate += text[j]

    return license_plate


def license_complies_format(text: str) -> bool:
    if len(text) != 8:
        return False

    if ((text[0] in string.ascii_uppercase or dict_int_to_char.keys()) and
            (text[1] in string.ascii_uppercase or dict_int_to_char.keys()) and
            (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or dict_char_to_int.keys()) and
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or dict_char_to_int.keys()) and
            (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or dict_char_to_int.keys()) and
            (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or dict_char_to_int.keys()) and
            (text[6] in string.ascii_uppercase or dict_int_to_char.keys()) and
            (text[7] in string.ascii_uppercase or dict_int_to_char.keys())):
        return True

    return False
