import requests

from interface.car_info import CarInfoResponse


class CarInfo:
    API_URL = "https://baza-gai.com.ua"
    API_KEY = "e1c4f104e5fb2a9efabf1d5835e0d0f5"

    def get_car_info(self, license_plate: str) -> CarInfoResponse:
        try:
            url = f"{self.API_URL}/nomer/{license_plate}"
            car_info = requests.get(url, headers={
                "Accept": "application/json",
                "X-Api-Key": self.API_KEY
            }).json()
            if 'error' in car_info:
                raise Exception(f'Не вдалось отримати інформацію з БД ДАІ по номеру {license_plate}')

            return CarInfoResponse(**car_info)
        except Exception as e:
            raise Exception(f'Не вдалось отримати інформацію з БД ДАІ по номеру {license_plate}')


