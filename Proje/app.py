import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from ultralytics import YOLO
import torchvision.transforms as transforms
import torch.nn as nn
import cv2

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32*30*30, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)

cnn_model = CNN()
cnn_model.load_state_dict(torch.load("cnn_model_final.pth", map_location="cpu"))
cnn_model.eval()

classes = ["Normal", "Stone"]

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

yolo_model = YOLO("best.pt")

root = tk.Tk()
root.title("Kidney Stone AI Detection")
root.geometry("850x650")
root.configure(bg="#1e1e1e")

img_label = tk.Label(root, bg="#1e1e1e")
img_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14), fg="white", bg="#1e1e1e")
result_label.pack()

def select_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img_cv = cv2.imread(file_path)
    if img_cv is None:
        return

    results = yolo_model(file_path)
    yolo_text = "YOLO: "

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            radius = int(max(x2 - x1, y2 - y1) / 2)
            cv2.circle(img_cv, (cx, cy), radius, (255, 0, 0), 2)

            name = yolo_model.names[cls]
            yolo_text += f"{name} %{conf*100:.1f} "

    img_pil = Image.open(file_path).convert("RGB")
    img_cnn = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        output = cnn_model(img_cnn)
        prob = torch.softmax(output, dim=1)

        normal_prob = prob[0][0].item()
        stone_prob = prob[0][1].item()

    cnn_text = f"CNN: Normal %{normal_prob*100:.1f} | Stone %{stone_prob*100:.1f}"

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_resized = img_pil.resize((450,450))
    img_tk = ImageTk.PhotoImage(img_resized)

    img_label.config(image=img_tk)
    img_label.image = img_tk

    result_label.config(text=f"{yolo_text}\n{cnn_text}")

btn = tk.Button(root, text="Resim Seç", command=select_image,
                bg="#4CAF50", fg="white", font=("Arial", 12))
btn.pack(pady=10)

root.mainloop()
