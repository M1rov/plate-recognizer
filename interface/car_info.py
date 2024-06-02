from pydantic import BaseModel


class CarRegion(BaseModel):
    name_ua: str


class OperationDescription(BaseModel):
    ua: str


class CarColor(BaseModel):
    ua: str


class CarOperation(BaseModel):
    is_last: bool
    registered_at: str
    operation: OperationDescription
    department: str
    color: CarColor
    address: str


class CarInfoResponse(BaseModel):
    digits: str
    vin: str
    vendor: str
    model: str
    model_year: int
    region: CarRegion
    photo_url: str
    is_stolen: bool
    operations: list[CarOperation]
