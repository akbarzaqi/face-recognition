import cv2
import os
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = input('\n masukan users id ==>  ')
print("\n [INFO] Inisialisasi kamera. lihat camera ...")
count = 0
while(True):
    ret, img = cam.read()
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        
        
       
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff 
      
    if k%256 == 27:
        print("Program selesai")
        break
    
    elif k%256 == 32:
        cv2.imwrite("dataset/User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
        print("Screenshot taken")
        count += 1
print("\n [INFO] Keluar dari program")
cam.release()
cv2.destroyAllWindows()