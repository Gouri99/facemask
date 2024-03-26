import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.utils import img_to_array,load_img
st.set_page_config(page_title="Mask Detection",page_icon="https://cdn-icons-png.freepik.com/256/3579/3579773.png")
facemodel=cv2.CascadeClassifier("face.xml")
#facemodel = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
maskmodel = load_model("mask.h5")
st.title("Face Mask Detection System")
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR27lRViAFaAqf44B2HWyKSOLe3sJl03xoNYQ&usqp=CAU")
choise=st.sidebar.selectbox("Menu",("Home","Image","URL","Web-Cam"))
if (choise=="Home"):
    st.image("https://assets.devfolio.co/hackathons/2c3a1c0ae6a44e2eb347210aa046599b/projects/c60a1b951d514d3ea6f93c5d8491f2a2/picplt85bfve.png")
    st.write("Face Mask Detection Sysem is a Computer Vision Machine Learning Application which can be accessed through IP cameras and can detect wheather the person is wearing a mask or not.")
elif(choise=="Image"):
    st.markdown('<center><h2>IMAGE DETECTION</h2></center>',unsafe_allow_html=True)
    file=st.file_uploader("Upload an Image")
    if file:
        b=file.getvalue()
        a=np.frombuffer(b,np.uint8)
        img=cv2.imdecode(a,cv2.IMREAD_COLOR)
        face=facemodel.detectMultiScale(img)
        for(x,y,l,w) in face:
            cv2.imwrite("temp.jpg",img[y:y+w,x:x+l])
            face_img=load_img("temp.jpg",target_size=(150,150,3))
            face_img=img_to_array(face_img)
            face_img=np.expand_dims(face_img,axis=0)
            pred=maskmodel.predict(face_img)[0][0]
            if(pred==1):
                cv2.rectangle(img,(x,y),(x+l,y+w),(0,0,255),8)
            else:
                cv2.rectangle(img,(x,y),(x+l,y+w),(0,255,0),8)
        st.image(img,channels='BGR')
elif(choise=='Web-Cam'):
    k=st.text_input("Enter 0 from Primary Camera or 1 for Secondary camera")
    btn=st.button("Start Camera")
    if btn:
        window=st.empty()
        k=int(k)
        vid=cv2.VideoCapture(k)
        btn2=st.button("Stop Camera")
        if(btn2):
            vid.release()
            st.experimental_rerun()
        #vid=cv2.VideoCapture(0) 
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                face=facemodel.detectMultiScale(frame)
                for(x,y,l,w) in face:
                    cv2.imwrite("temp.jpg",frame[y:y+w,x:x+l])
                    face_img=load_img("temp.jpg",target_size=(150,150,3))
                    face_img=img_to_array(face_img)
                    face_img=np.expand_dims(face_img,axis=0)
                    pred=maskmodel.predict(face_img)[0][0]
                    if(pred==1):
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),8)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),8)
                window.image(frame,channels="BGR")
elif(choise=='URL'):
    k=st.text_input("Enter URL for video")
    btn=st.button("Start Camera")
    if btn:
        window=st.empty()
        vid=cv2.VideoCapture(k)
        btn2=st.button("Stop Camera")
        if(btn2):
            vid.release()
            st.experimental_rerun()
        #vid=cv2.VideoCapture(0) 
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                face=facemodel.detectMultiScale(frame)
                for(x,y,l,w) in face:
                    cv2.imwrite("temp.jpg",frame[y:y+w,x:x+l])
                    face_img=load_img("temp.jpg",target_size=(150,150,3))
                    face_img=img_to_array(face_img)
                    face_img=np.expand_dims(face_img,axis=0)
                    pred=maskmodel.predict(face_img)[0][0]
                    if(pred==1):
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),8)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),8)
                window.image(frame,channels="BGR")
