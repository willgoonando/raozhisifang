import streamlit as st

st.title("首页")


#字典
pages ={
#一级菜单
    'Hough变换':
    [
         st.Page('Page/line_page.py',title='直线检测'),
          st.Page('Page/circle_page.py',title='圆形检测')

    ],

        '边界检测':
    [
        st.Page('Page/edgedet_page.py',title='边界检测'),
        #st.Page('circle_page.py',title='圆形检测')

    ],
    '视频':[st.Page('Page/video_show.py',title='视频播放')],
    '人脸识别':[st.Page('Page/faceTest.py',title='人脸识别')],
    '目标检测':[st.Page('Page/yolo_page.py',title='YOLOv8目标检测')],
    '图像处理':[st.Page('Page/image_process_page.py',title='图像处理')]
  } 

#创建导航栏
pg = st.navigation(pages)
pg.run()