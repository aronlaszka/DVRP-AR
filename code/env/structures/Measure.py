from common.general import convert_sec_to_hh_mm_ss


class Measure:
    def __init__(self, value=0, unit=""):
        self._value = value
        self._unit = unit
        self._init()

    def _init(self):
        raise NotImplementedError

    def get(self, unit=""):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            new_value = self._value + other
        elif isinstance(other, type(self)):
            new_value = self._value + other.get(self._unit)
        else:
            raise ValueError(f"Invalid type to add {type(other)}")
        return type(self)(new_value, unit=self._unit)

    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            new_value = self._value - other
        elif isinstance(other, type(self)):
            new_value = self._value - other.get(self._unit)
        else:
            raise ValueError(f"Invalid type to subtract {type(other)}")
        return type(self)(new_value, unit=self._unit)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            new_value = self._value * other
        elif isinstance(other, type(self)):
            new_value = self._value * other.get(self._unit)
        else:
            raise ValueError(f"Invalid type to add {type(other)}")
        return type(self)(new_value, unit=self._unit)

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            if int(other) != 0:
                new_value = self._value / other
            else:
                # temporary fix for zero-division error
                new_value = 0
        elif isinstance(other, type(self)):
            if int(other.get(self._unit)) != 0:
                new_value = self._value / other.get(self._unit)
            else:
                # temporary fix for zero-division error
                new_value = 0
        else:
            raise ValueError(f"Invalid type to add {type(other)}")
        return type(self)(new_value, unit=self._unit)

    def __round__(self, n=None):
        return round(self._value, n)


class Distance(Measure):
    def __init__(self, value=0, unit="m"):
        self.meters = 0
        self.miles = 0
        super(Distance, self).__init__(value, unit)

    def _init(self):
        if self._unit == "m":
            self.meters = self._value
            self.miles = round(self._value / 1609.344, 3)
        elif self._unit == "mi":
            self.meters = self._value * 1609.344
            self.miles = self._value

    def get(self, unit="m"):
        if unit == self._unit:
            return self._value
        elif unit == "m":
            return self.meters
        elif unit == "mi":
            return self.miles

    def __str__(self):
        return str(self.meters)


class Duration(Measure):
    def __init__(self, value=0, unit="s"):
        self.seconds = 0
        self.minutes = 0
        self.hours = 0
        super(Duration, self).__init__(value, unit)

    def _init(self):
        if self._unit == "s":
            self.seconds = self._value
            self.minutes = self._value / 60.0
            self.hours = self._value / 3600.0
        elif self._unit == "m":
            self.seconds = self._value * 60.0
            self.minutes = self._value
            self.hours = self._value / 60.0
        elif self._unit == "h":
            self.seconds = self._value * 3600.0
            self.minutes = self._value * 60.0
            self.hours = self._value
        self.seconds = int(self.seconds)

    def get(self, unit="s"):
        if unit == self._unit:
            return self._value
        elif unit == "s":
            return self.seconds
        elif unit == "m":
            return self.minutes
        elif unit == "h":
            return self.hours

    def __str__(self):
        return str(self.seconds)

    def hhmmss(self):
        return convert_sec_to_hh_mm_ss(self.seconds)
