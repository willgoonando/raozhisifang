import cv2
import face_recognition
import numpy as np
# 读取图片
im = cv2.imread('./images/face1.jpg')

# 人脸检测
face_locations = face_recognition.face_locations(im, number_of_times_to_upsample=1, model='hog')
print(f"人脸检测结果: {face_locations}")

# 人脸特征值
face_feature = face_recognition.face_encodings(im, face_locations)
print(f"人脸检测的特征: {face_feature}, 特征的维度: {face_feature[0].shape}")

# 人脸标注点
face_landmarks = face_recognition.face_landmarks(im, face_locations)
f1 = face_landmarks[0]
print(f1['left_eye'])

#汇智人脸检测标注的信息
face_lip = face_landmarks[0]['top_lip']
#绘制嘴的轮廓
cv2.polylines(im,[np.array(face_lip)],True,(0,0,255),2)

#填充嘴唇
cv2.fillPoly(im,[np.array(face_lip)],(0,0,255))
face_lip = face_landmarks[0]['bottom_lip']
cv2.fillPoly(im,[np.array(face_lip)],(0,0,255))

for face in face_landmarks:
    #face====字典
    for k,v in face.items():
        print(f'键值：{k},键值对应的·value{v}')
        for pt in v:
            #pt是标注中的每一个点
            #画出人脸
            cv2.circle(im,pt,2,(255,0,0),1)

# 遍历检测到的人脸位置信息，获取坐标并绘制人脸框
for face in face_locations:
    top, right, bottom, left = face
    left_top = (left, top)
    right_bottom = (right, bottom)
    print(f"左上角: {left_top}, 右下角: {right_bottom}")
    # 画出人脸框，参数分别为 图像、左上角坐标、右下角坐标、颜色(BGR)、线条宽度
    cv2.rectangle(im, left_top, right_bottom, (255, 0, 0), 2) 

# 显示处理后的图像
cv2.imshow('face', im)
cv2.waitKey(0)
cv2.destroyAllWindows()