import re
import string

dict_char_to_int = {
    'O': '0',
    'I': '1',
    'J': '3',
    'A': '4',
    'G': '6',
    'S': '5',
    'B': '8'
}

dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '3': 'J',
    '4': 'A',
    '6': 'G',
    '5': 'S',
    '8': 'B'
}


class PlateValidation:
    @staticmethod
    def format_license_plate(text: str) -> str:
        cleaned_text = re.sub(r'[^A-Z0-9]', '', text)

        if len(cleaned_text) > 8:
            cleaned_text = cleaned_text[-8:]

        license_plate = ''
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_char_to_int, 3: dict_char_to_int,
                   4: dict_char_to_int,
                   5: dict_char_to_int, 6: dict_int_to_char, 7: dict_int_to_char}
        for j in [0, 1, 2, 3, 4, 5, 6, 7]:
            if cleaned_text[j] in mapping[j].keys():
                license_plate += mapping[j][cleaned_text[j]]
            else:
                license_plate += cleaned_text[j]

        return license_plate

    @staticmethod
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