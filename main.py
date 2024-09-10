from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    # Load the YOLOv9c model from pretrained weights
    model = YOLO(model="./models/runs/train/exp/weights/best.pt")
    
    # Read the image using OpenCV
    img = cv2.imread("./datasets_yolov9/test/images/img004_20201027_Kaizu_TCA422_515x515_png.rf.99667cd29d2038060c83c32f5a641244.jpg")
    
    # Display model information (optional)
    model.info()
    
    # Use the model to process the image
    results = model.detect(img, conf=0.2,save=True)
  