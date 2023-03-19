import cv2
import numpy as np

from utils import load_model_resnet152, draw_str
from constants import img_height, img_width, class_names, num_classes

if __name__ == '__main__':
    model = load_model_resnet152()
    
    video_path = 'videos/cars_video_3spc.mp4'
    
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        frame = cv2.resize(frame, (800, 800), cv2.INTER_CUBIC)
        bgr_img = cv2.resize(frame, (img_width, img_height), cv2.INTER_CUBIC)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)
        preds = model.predict(rgb_img)
        prob = np.max(preds)
        class_index = np.argmax(preds)
        text = 'Unknown'
        if class_index >= 0 and class_index < num_classes:
            text = '{}: {}'.format(
                class_names[class_index][0][0], round(float(prob), 4))
        
        draw_str(frame=frame, position=(10, 50), text=text)
        cv2.imshow('Car classificaion', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()