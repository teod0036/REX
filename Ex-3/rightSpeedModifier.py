

def calc_rightSpeedModifier() -> list[int]:
    def f(speed: int) -> int:
        if 42 <= speed < 52:
            return 3
        elif 52 <= speed < 63:
            return 3
        elif (l := 63) <= speed < (r := 74):
            a, b = 3, 5
            t = (speed - l) / (r - l)
            return round(a * (1 - t) + b * t)
        elif (l := 74) <= speed < (r := 85):
            a, b = 5, 6
            t = (speed - l) / (r - l)
            return round(a * (1 - t) + b * t)
        elif (l := 85) <= speed < (r := 96):
            a, b = 6, 7
            t = (speed - l) / (r - l)
            return round(a * (1 - t) + b * t)
        elif 96 <= speed < 107:
            return 7
        elif (l := 107) <= speed < (r := 118):
            a, b = 7, 8
            t = (speed - l) / (r - l)
            return round(a * (1 - t) + b * t)
        elif (l := 118) <= speed < (r := 128):
            a, b = 8, 13
            t = (speed - l) / (r - l)
            return round(a * (1 - t) + b * t)
        else:
            return 1

    return [-f(i) for i in range(0, 127 + 1)]

rightSpeedModifier = calc_rightSpeedModifier()
