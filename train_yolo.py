from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=r"C:\Users\Melih\Desktop\derin öğrenme\dataset\data.yaml",
    epochs=5,     
    imgsz=320,       
    batch=8,        
    workers=2,      
    device="cpu"     
)