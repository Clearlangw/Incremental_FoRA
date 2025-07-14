import cv2

# 读取JPEG图像
image = cv2.imread('M1401_001050.jpg')

# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 保存灰度图像为BMP格式
cv2.imwrite('M1401_001050.bmp', gray_image)
