import streamlit as st
import cv2
import face_recognition
import numpy as np

st.title("人脸识别与嘴唇颜色调整")
st.write("上传图片后可检测人脸并调整嘴唇颜色")

# 上传图片
uploaded_file = st.file_uploader("请上传一张图片", type=["jpg", "jpeg", "png"])

def process_image(im, r, g, b):
    """处理图像：检测人脸并填充嘴唇颜色"""
    # 转换为RGB（face_recognition需要RGB格式）
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # 人脸检测
    face_locations = face_recognition.face_locations(rgb_im, model='hog')
    face_landmarks = face_recognition.face_landmarks(rgb_im, face_locations)
    
    if face_landmarks:
        # 复制原图用于绘制
        im_draw = im.copy()
        
        # 处理每个人脸的嘴唇
        for landmarks in face_landmarks:
            try:
                # 上嘴唇
                top_lip = landmarks['top_lip']
                cv2.fillPoly(im_draw, [np.array(top_lip)], (b, g, r))  # BGR格式
                
                # 下嘴唇
                bottom_lip = landmarks['bottom_lip']
                cv2.fillPoly(im_draw, [np.array(bottom_lip)], (b, g, r))  # BGR格式
                
                # 绘制人脸框
                top, right, bottom, left = face_locations[face_landmarks.index(landmarks)]
                cv2.rectangle(im_draw, (left, top), (right, bottom), (0, 255, 0), 2)
                
            except KeyError:
                st.warning("未检测到完整嘴唇特征点")
        
        return im_draw, face_locations
    else:
        return None, []

if uploaded_file is not None:
    # 读取图片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    im = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 显示原图
    st.subheader("原始图像")
    st.image(im, channels="BGR", use_column_width=True)
    
    # RGB参数调节滑块
    st.sidebar.header("颜色调整参数")
    r = st.sidebar.slider("红色通道 (R)", 0, 255, 255)
    g = st.sidebar.slider("绿色通道 (G)", 0, 255, 0)
    b = st.sidebar.slider("蓝色通道 (B)", 0, 255, 0)
    
    if st.button("开始处理"):
        # 处理图像
        result_im, face_locations = process_image(im, r, g, b)
        
        if result_im is not None:
            # 显示结果
            st.subheader("处理结果")
            st.image(result_im, channels="BGR", use_column_width=True)
            
            # 显示人脸检测信息
            st.write(f"检测到 {len(face_locations)} 个人脸")
            for i, (top, right, bottom, left) in enumerate(face_locations, 1):
                st.write(f"人脸 {i} 坐标：左上角({left}, {top}), 右下角({right}, {bottom})")
        else:
            st.warning("未检测到人脸或嘴唇特征点")
else:
    st.info("请上传图片以开始处理")