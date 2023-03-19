# necessary libs
import cv2
import numpy as np
from utils import *
import time

# car classification
# from constants import img_height, img_width, class_names, num_classes

# license plate recognition
import pytesseract
from lib_detection import detect_lp, im2single





if __name__ == '__main__':
  # resnet152_model = load_model_resnet152()
  lpr_model = load_model_wpod_net()
  
  img_path = "images/car9.jpg"
  video_path = 'videos/cars_video_3spc.mp4'
  # cap = cv2.VideoCapture(video_path)
  cap = cv2.imread(img_path)
  
  while True:
    # startTime = time.time()
    
    # ret, frame = cap.read()
    ret, frame = True, cap

    # car classification
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break
    
    # Dmax, Dmin = 800, 800
    # frame = cv2.resize(frame, (Dmax, Dmin), cv2.INTER_CUBIC)
    # bgr_img = cv2.resize(frame, (img_width, img_height), cv2.INTER_CUBIC)
    # rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    # rgb_img = np.expand_dims(rgb_img, 0)
    # preds = resnet152_model.predict(rgb_img)
    # prob = np.max(preds)
    # class_index = np.argmax(preds)
    # text = 'Unknown'
    # if class_index >= 0 and class_index < num_classes:
    #   text = '{}: {}'.format(
    #     class_names[class_index][0][0], round(float(prob), 4))
    #   draw_str(frame=frame, position=(10, 100), text=text)
      

    # lpr
    # Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
    Dmax, Dmin = 608, 288
    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(frame.shape[:2])) / min(frame.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, lp_type = detect_lp(lpr_model, im2single(frame), bound_dim, lp_threshold=0.5)
    
    if (len(LpImg)):
      # Chuyen doi anh bien so
      LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
      # Chuyen anh bien so ve gray
      gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)
      # cv2.imshow("Anh bien so sau chuyen xam", gray)
      # Ap dung threshold de phan tach so va nen
      binary = cv2.threshold(gray, 127, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
      # cv2.imshow("Anh bien so sau threshold", binary)
      # Nhan dien bien so. Cau hinh --psm 7 la de nhan dien 1 line only
      text = pytesseract.image_to_string(binary, lang="eng", config="--psm 7")

      # Viet bien so len anh
      # cv2.putText(frame,fine_tune(text),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_8, thickness=2)
      draw_str(frame=frame, position=(10, 50), text=text)

    cv2.imshow('Car classificaion', frame)
    
    # endTime = time.time()
    # print(f'run time = {format(endTime - startTime)}')
    
    if cv2.waitKey(1) == ord('q'):
      break
