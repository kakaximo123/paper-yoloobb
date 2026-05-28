import math

from ultralytics import YOLO

# 加载预训练的 YOLOv8n 模型
model = YOLO(
    r"D:\examples_project\python projects\self-enhance-all\ultralytics-main-3\ultralytics-main\runs\obb\train65\weights\best.pt"
)
# model = YOLO('yolov8n.pt')
# 定义图像文件的路径
# source = (r'"D:\examples_project\python projects\ultralytics-main\ultralytics-main\camera\photo\capture_20250911_150809.jpg"') #更改为自己的图片路径
# 运行推理，并附加参数
# model.predict(source, save=True)
results = model.predict(source=r"D:\研究生\会议论文\2222.png", task="obb", save=True)
print(results)

for r in results:
    if r.obb is not None and len(r.obb.xywhr) > 0:
        # xywhr 是 [cx, cy, w, h, angle]
        for box in r.obb.xywhr.cpu().numpy():
            cx, cy, w, h, angle = box
            print(f"cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}, angle={math.degrees(angle):.1f}°")
    else:
        print("No OBB detected")
