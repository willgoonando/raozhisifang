import cv2
import face_recognition
import numpy as np

# 读取图片
image_path = './images/face1.jpg'
im = cv2.imread(image_path)

if im is None:
    print(f"无法加载图像: {image_path}")
    exit()

# 复制原始图像用于重置
original_im = im.copy()

# 将BGR转换为RGB (face_recognition使用RGB)
rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# 人脸检测
face_locations = face_recognition.face_locations(rgb_im, number_of_times_to_upsample=1, model='hog')
print(f"人脸检测结果: {face_locations}")

if not face_locations:
    print("未检测到人脸")
    exit()

# 人脸特征值
face_feature = face_recognition.face_encodings(rgb_im, face_locations)
print(f"人脸检测的特征: {face_feature}, 特征的维度: {face_feature[0].shape}")

# 人脸标注点
face_landmarks = face_recognition.face_landmarks(rgb_im, face_locations)
print(f"检测到 {len(face_landmarks)} 个人脸的标注点")

# 创建窗口
cv2.namedWindow('face')

# 初始值
current_r = 0
current_g = 0
current_b = 0

def update_lips():
    global im
    # 重置图像
    im = original_im.copy()
    
    if face_landmarks:
        # 处理所有检测到的人脸
        for landmarks in face_landmarks:
            try:
                # 填充上嘴唇
                top_lip = landmarks['top_lip']
                cv2.fillPoly(im, [np.array(top_lip)], (current_b, current_g, current_r))  # BGR模式
                
                # 填充下嘴唇
                bottom_lip = landmarks['bottom_lip']
                cv2.fillPoly(im, [np.array(bottom_lip)], (current_b, current_g, current_r))  # BGR模式
            except (IndexError, KeyError) as e:
                print(f"处理标注点时出错: {e}")
    
    # 更新显示
    cv2.imshow('face', im)

def fun_r(val):
    global current_r
    current_r = val
    update_lips()

def fun_g(val):
    global current_g
    current_g = val
    update_lips()

def fun_b(val):
    global current_b
    current_b = val
    update_lips()

# 创建三个滑块分别控制RGB
cv2.createTrackbar('R', 'face', 0, 255, fun_r)
cv2.createTrackbar('G', 'face', 0, 255, fun_g)
cv2.createTrackbar('B', 'face', 0, 255, fun_b)

# 初始化显示
update_lips()

# 等待按键
cv2.waitKey(0)
cv2.destroyAllWindows()