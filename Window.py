import tkinter as tk
import cv2
from tkinter import Canvas
import torch
import torchvision
from Resnet import Residual
from PIL import Image, ImageTk

model_path = "full_model199.pth"
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
          'W', 'X', 'Y', 'Z']

app = tk.Tk()
app.title("手势识别")

canvas = Canvas(app, width=224, height=224)
canvas.pack()

# 创建一个标签Canvas用于绘制标签文本
label_canvas = Canvas(app, width=224, height=30)
label_canvas.pack()

cap = cv2.VideoCapture(0)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
net = torch.load(model_path)
print(net)
net.eval()
net.to(device)

# 初始化标签文本
label_canvas.create_text(10, 10, text="识别结果（top5）: ", anchor=tk.NW)
label_text = label_canvas.create_text(100, 10, text="", anchor=tk.NW)

# 初始化PhotoImage对象
photo = None

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
top_left_x = (width - 224) // 2
top_left_y = (height - 224) // 2


def classify_image():
    global photo

    ret, frame = cap.read()
    frame = frame[top_left_y:top_left_y + 224, top_left_x:top_left_x + 224]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 调整图像大小以匹配模型的输入尺寸，并进行预处理
    img = Image.fromarray(frame)
    img = transform(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        output = net(img)
    # _, predicted = torch.max(output, 1)
    # label = labels[predicted]
    topk_values, topk_indices = torch.topk(output, k=5)
    topk_indices = topk_indices.squeeze(0)
    label = [labels[i] for i in topk_indices]

    # 更新标签文本
    label_canvas.itemconfig(label_text, text=label)

    # 使用Pillow中的ImageTk将图像转换为PhotoImage
    pil_image = Image.fromarray(frame)
    photo = ImageTk.PhotoImage(image=pil_image)

    # 在Canvas上显示图像
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    canvas.update()
    app.after(10, classify_image)


classify_image()

app.mainloop()

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
