'''
import requests  # requests 모듈 임포트

def download_file(file_url, save_path):
    with open(save_path, 'wb') as f: # 저장할 파일을 바이너리 쓰기 모드로 열기
        response = requests.get(file_url) # 파일 URL에 GET 요청 보내기
        f.write(response.content) # 응답의 내용을 파일에 쓰기

# URL과 저장 경로 변수를 지정합니다.

url = f'https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B/NR016/FD/data?date=202210272350&authKey=JJA3FaboS9KQNxWm6DvS'
save_file_path = 'F:/INKLE/satelite/output_file.nc'

# 파일 다운로드 함수를 호출합니다.
download_file(url, save_file_path)
'''

import requests, os #API 요청, 파일 시스템 작업업
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def download_file(url, save_path, auth_key):
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)  # 디렉토리 생성

    # 요청 헤더에 인증키 추가
    headers = {
        "Authorization": f"Bearer JJA3FaboS9KQNxWm6DvS"  # 인증 방식에 맞게 설정
    }

    # API 요청
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()  # HTTP 에러 확인

    # 파일 저장
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


# URL과 경로 설정
url = "https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B/NR016/KO/data?date=202210272350&authKey=JJA3FaboS9KQNxWm6DvSBQ"
auth_key = "JJA3FaboS9KQNxWm6DvS"
save_file_path = r'F:/INKLE/satelite/output_file.nc'

# 파일 다운로드 호출
download_file(url, save_file_path, auth_key)
