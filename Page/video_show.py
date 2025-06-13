import cv2
import streamlit as st
#处理临时文件库
import tempfile


st.info('视频播放')



#st.video(r'C:\Users\Administrator\Desktop\5.30课程\tst.mp4')

file_uploader = st.file_uploader('上传视频',type=["mp4"])

if file_uploader:
    #初始化临时文件的对象
    temp = tempfile.TemporaryFile()
    #将上传的视频文件写入临时文件
    temp.write(file_uploader.read())
    #使用opencv的处理逻辑处理视频
    cap = cv2.VideoCapture(temp.name)
    frame_shw = st.empty()  #占位符
    if cap.isOpened():
    #判断视频对象
        while True: 
            ret,frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                #读取成功,在视频上画圆
                cv2.circle(frame, (20, 20), 20, (255, 255, 0), 2)
                #cv2.imshow("video",frame)
                #在页面中显示视频帧
                frame_shw.image(frame)
                cv2.waitKey(20)





            else:
                print("读取视频失败!")
                break
    else:
        print('cap对象准备失败！')

    #3.销毁资源
    cap.release()
    cv2.destroyAllWindows()
