import streamlit as st
import cv2
import numpy as np
st.title("边界检测")

files =  st.file_uploader('图形上传',type=["jpg","png","jpeg"])

#将页面拆分为两个布局
col1,col2 = st.columns(2)
if files is not None:
    #通过getvalue获得的数据是二值文件
    values  = files.getvalue()
    #st.write(values)
    #转换为opencv可以识别的
    cv2_img = cv2.imdecode(np.frombuffer(values,np.uint8),cv2.IMREAD_COLOR)

    #将左边布局显示原始图像
    with col1:
        st.image(cv2_img,'原始图像')


    #业务逻辑(最好单独放一个utils.py文件里面处理)
    def cannyDet(img):
        edge = cv2.Canny(cv2.cvtColor(cv2_img,cv2.COLOR_BGR2GRAY),100,200)
        return edge
    #边界检测
    img_gary = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(img_gary,100,200)


    with col2:
        st.image(cannyDet(cv2_img),'检测结果')
    #image自动处理了二值文件
    st.image(cv2_img)
