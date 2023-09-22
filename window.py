import os
import random
import tkinter as tk
import cv2
from tkinter import Canvas, messagebox
import torch
import torchvision
from PIL import Image, ImageTk, ImageDraw, ImageFont
from Resnet import Residual  # 残差块

MODEL_PATH = "full_model199.pth"  # 训练好的模型
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
          'W', 'X', 'Y', 'Z']

# 创建窗口
app = tk.Tk()
app.title("Gesture Recognition")

# 获取显示器长宽
SCREEN_WIDTH, SCREEN_HEIGHT = app.winfo_screenwidth(), app.winfo_screenheight()

# 摄像头
cap = cv2.VideoCapture(0)

# 摄像头分辨率
CAMERA_WIDTH, CAMERA_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置窗口位置
WINDOW_GEOMETRY = '%dx%d+%d+%d' % (CAMERA_WIDTH + 100, CAMERA_HEIGHT,
                                   (SCREEN_WIDTH - CAMERA_WIDTH) // 2, (SCREEN_HEIGHT - CAMERA_HEIGHT) // 2)
app.geometry(WINDOW_GEOMETRY)
app.resizable(False, False)  # 窗口不可移动

# interest区域
TOP_LEFT_X = (CAMERA_WIDTH - 224) // 2
TOP_LEFT_Y = (CAMERA_HEIGHT - 224) // 2

# 创建画布
canvas = Canvas(app, width=CAMERA_WIDTH + 100, height=CAMERA_HEIGHT)
canvas.pack(side=tk.LEFT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = torchvision.transforms.Compose([  # 图像转换
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载模型
gesture_model = torch.load(MODEL_PATH)
gesture_model.eval()
gesture_model.to(DEVICE)

photo = None  # 要处理的图像
sample_img = None  # 样例图片
match_times = 0  # 匹配成功的次数
all_pred_times = 0  # 尝试匹配的穿刺术
rectangle_color = (255, 255, 255)  # 方框和状态文本的颜色
status_text = "识别中……"  # 状态文本


# 获取样例图片和标签
def get_sample_img():
    num = random.randint(0, 25)
    path = os.path.join("sample", LABELS[num] + ".jpg")
    return ImageTk.PhotoImage(Image.open(path)), LABELS[num]


# 显示样例图片
def update_sample_img():
    global sample_img
    sample_img, real = get_sample_img()
    canvas.create_image(100, CAMERA_HEIGHT // 2, anchor=tk.E, image=sample_img)
    return real


def window():
    global photo, all_pred_times, match_times, real_label, rectangle_color, status_text
    ret, frame = cap.read()  # 捕获图像
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def predict(frame_image, match_times_now):
        # 预测手势
        frame_image = frame_image[TOP_LEFT_Y:TOP_LEFT_Y + 224, TOP_LEFT_X:TOP_LEFT_X + 224]  # 截取图像
        img = Image.fromarray(frame_image)
        img = TRANSFORM(img).unsqueeze(0)
        img = img.to(DEVICE)
        with torch.no_grad():
            output = gesture_model(img)
        # 使用概率最高的5个标签
        topk_values, topk_indices = torch.topk(output, k=5)
        topk_indices = topk_indices.squeeze(0)
        label = [LABELS[i] for i in topk_indices]
        print(f"Top 5 结果: {label}")
        print(f"真实标签：: {real_label}")
        for i in label:
            if i == real_label:
                match_times_now += 1  # 每次匹配成功，次数+1
        return match_times_now

    def process_match():
        # 处理匹配结果
        if match_times >= int(30 * 0.6):
            new_rectangle_color = (144, 238, 144)
            text = "识别成功"
            messagebox.showinfo("识别结果", "识别成功")
        else:
            new_rectangle_color = (255, 0, 0)
            text = "识别失败"
            messagebox.showinfo("识别结果", "识别失败")
        return new_rectangle_color, text

    def process(pred_times, match_times_now, rect_color):
        # 更新状态，根据总识别次数有不同的状态，<30为识别次数，30-60为展示识别结果，60-80等待
        global status_text, real_label
        if pred_times < 30:
            status_text = "识别中……"
            match_times_now = predict(frame, match_times_now)
        elif pred_times == 30:
            rect_color, status_text = process_match()
        elif pred_times == 60:
            real_label = update_sample_img()
            status_text = "等待中……"
            match_times_now = 0
            rect_color = (255, 255, 255)
        elif pred_times == 80:
            pred_times = 0
        return pred_times, match_times_now, rect_color

    # 更新总匹配次数，匹配成功次数，矩形颜色
    all_pred_times, match_times, rectangle_color = process(all_pred_times, match_times, rectangle_color)

    def add_rectangle():
        # 添加矩形
        cv2.rectangle(frame, (TOP_LEFT_X, TOP_LEFT_Y), (TOP_LEFT_X + 224, TOP_LEFT_Y + 224), rectangle_color, 2)

    def add_hint():
        # 添加矩形下方的提示文本
        font_size = 28
        font = ImageFont.truetype("Deng.ttf", font_size)
        text = "在框内做出给定的手势"
        text_x = TOP_LEFT_X + (224 - font_size * len(text)) // 2
        text_y = TOP_LEFT_Y + 224 + 20
        text_position = (text_x, text_y)
        draw = ImageDraw.Draw(pil_image)
        draw.text(text_position, text, fill=(255, 255, 255), font=font)

    def add_status():
        # 添加矩形上方的状态文本
        font_size = 28
        font = ImageFont.truetype("Deng.ttf", font_size)
        text_x = TOP_LEFT_X + (224 - font_size * len(status_text)) // 2
        text_y = TOP_LEFT_Y - 40
        text_position = (text_x, text_y)
        draw = ImageDraw.Draw(pil_image)
        draw.text(text_position, status_text, fill=rectangle_color, font=font)

    add_rectangle()
    pil_image = Image.fromarray(frame)
    add_hint()
    add_status()

    # 显示摄像头捕获的图像
    photo = ImageTk.PhotoImage(image=pil_image)
    canvas.create_image(100, 0, image=photo, anchor=tk.NW)

    all_pred_times += 1

    app.after(50, window)


canvas.update()

real_label = update_sample_img()  # 图片的真实标签
print(real_label)

window()


def exit_app():
    cap.release()
    cv2.destroyAllWindows()
    app.destroy()


app.protocol("WM_DELETE_WINDOW", exit_app)

app.mainloop()
