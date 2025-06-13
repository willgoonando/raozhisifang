import streamlit as st
import cv2
import numpy as np
from PIL import Image
st.title("直线检测")


def detect_lines(image, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10):
    """
    使用HoughLinesP检测图像中的直线
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Canny边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 检测直线
    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=minLineLength,
        maxLineGap=maxLineGap
    )
    
    return lines

st.title("直线检测应用")
st.write("上传一张图片，程序将自动检测其中的直线")
    
# 上传图片
uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])
    
if uploaded_file is not None:
    # 读取图片
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
    # 显示原图
    st.subheader("原始图片")
    st.image(image, use_column_width=True)
        
    # 参数设置滑块
    st.sidebar.header("参数设置")
    rho = st.sidebar.slider("距离分辨率 (像素)", 1, 10, 1)
    theta = st.sidebar.slider("角度分辨率 (弧度)", 0.01, np.pi/180, np.pi/180, 0.01)
    threshold = st.sidebar.slider("累加器阈值", 10, 500, 100)
    minLineLength = st.sidebar.slider("最小线段长度", 10, 500, 50)  # 修改默认值为50
    maxLineGap = st.sidebar.slider("最大线段间隙", 1, 100, 20)  # 修改默认值为20
        
    if st.button("检测直线"):
        # 检测直线
        lines = detect_lines(img_cv, rho, theta, threshold, minLineLength, maxLineGap)
            
        # 复制原图用于标记
        img_marked = img_cv.copy()
            
        if lines is not None:
            # 标记直线
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_marked, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
            # 计算每条直线的长度和角度
            line_data = []
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                line_data.append({
                        "直线序号": i+1,
                        "起点坐标": f"({x1}, {y1})",
                        "终点坐标": f"({x2}, {y2})",
                        "长度(像素)": f"{length:.2f}",
                        "角度(度)": f"{angle:.2f}"
                })
                
            # 转换回RGB显示
            img_result = cv2.cvtColor(img_marked, cv2.COLOR_BGR2RGB)
                
            # 显示结果
            st.subheader("检测结果")
            st.image(img_result, use_column_width=True)
                
            # 显示检测到的直线统计信息
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"成功检测到 {len(lines)} 条直线")
            with col2:
                avg_length = np.mean([float(line["长度(像素)"]) for line in line_data])
                st.info(f"平均长度: {avg_length:.2f} 像素")
                
            # 显示直线端点坐标表格
            st.subheader("直线详细信息")
            st.dataframe(line_data)
        else:
            st.warning("未检测到直线，请尝试调整参数")

