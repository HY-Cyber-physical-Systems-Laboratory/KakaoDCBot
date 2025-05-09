import cv2
import numpy as np
import base64
import requests
from io import BytesIO
from PIL import Image

# 1. API에서 데이터 받아오기
url = "http://192.168.106.132:8080/api/mapdata"
print(123)
response = requests.get(url)

print(123)
data = response.json()

print(123)
# 2. base64 문자열 추출 및 디코딩
base64_str = data['map_data'].split(',')[1]  # "data:image/jpeg;base64,..." 형식이므로 쉼표 이후 추출
img_data = base64.b64decode(base64_str)

print(123)
# 3. OpenCV 이미지로 변환
image = Image.open(BytesIO(img_data)).convert('RGB')
image_np = np.array(image)
image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

image_cv = cv2.flip(image_cv, 0)

print(123)
# 4. 그레이스케일 변환
gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

# 5. 흐리게 처리
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]

not_binary = cv2.bitwise_not(binary)  #// make not binary image


erosion_image = cv2.erode(not_binary, (5, 5), iterations=1)  #// make erosion image
dilation_image = cv2.dilate(not_binary, (5, 5), iterations=10)  #// make dilation image

not_distance = cv2.bitwise_not(dilation_image)  #// make not distance image
# 6. 엣지 검출

cv2.imshow('Detected Lines', dilation_image)
cv2.waitKey(0)
# 7. 허프 선 검출
lines = cv2.HoughLinesP(dilation_image, 1, np.pi / 180, threshold=50, minLineLength=15, maxLineGap=1)

# 8. 직선 그리기
output = image_cv.copy()
if lines is not None:
    for line in lines:
        x, y, xX, yY = line[0]
        x1, y1, x2, y2 = line[0]
        a =  (y2 - y1)/(x2 - x1)
        rad_a = np.arctan(a)
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        rad_orth = rad_a + np.pi / 2
        midpoint = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        distance = (((line_length / 2) ** 2 / 3) ** 0.5) * 2
        
        x1 = int(midpoint[0] + distance * np.cos(rad_orth))
        y1 = int(midpoint[1] + distance * np.sin(rad_orth))
        
        dot1 = np.array([x1, y1])

        x2 = int(midpoint[0] - distance * np.cos(rad_orth))
        y2 = int(midpoint[1] - distance * np.sin(rad_orth))

        dot2 = np.array([x2, y2])
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.line(output, (x, y), (xX, yY), (255, 0, 0), 2)
        cv2.circle(output, (x1, y1), 1, (0, 0, 255), -1)
        cv2.circle(output, (x2, y2), 1, (0, 0, 255), -1)
        
        #cv2.putText(output, f"({x1}, {y1})", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 9. 결과 보여주기 또는 저장'

print(123)
cv2.imshow('Detected Lines', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 또는 저장하고 싶으면 아래 코드 사용:
# cv2.imwrite("output_with_lines.jpg", output)
