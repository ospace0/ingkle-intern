class CoordRanger:
    def __init__(self):
        self.size_info = {
            "05": {
                "x_start": 1430, "x_end": 2655,
                "y_start": 1545, "y_end": 2883,
                "image_width": 3600,
                "image_height": 3600,
                "x_min": -899750,
                "x_max": 899750,
                "y_min": -899750,
                "y_max": 899750,
            },
            "10": {
                "x_start": 715, "x_end": 1327,
                "y_start": 772, "y_end": 1441,
                "image_width": 1800,
                "image_height": 1800,
                "x_min": -899500,
                "x_max": 899500,
                "y_min": -899500,
                "y_max": 899500,
            },
            "20": {
                "x_start": 357, "x_end": 663,
                "y_start": 410, "y_end": 720,
                "image_width": 900,
                "image_height": 900,
                "x_min": -899000,
                "x_max": 899000,
                "y_min": -899000,
                "y_max": 899000,
            }
        }
        self.lcc_params = {
            "lat_1": 30.0,   # 표준 평행선 1
            "lat_2": 60.0,   # 표준 평행선 2
            "lat_0": 38.0,   # 원점 위도
            "lon_0": 126.0   # 중심 자오선
        }
