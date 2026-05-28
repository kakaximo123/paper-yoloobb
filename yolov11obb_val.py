from ultralytics import YOLO
import torch


def main():
    # ------------------------------------------------------------------
    # 1. 设置模型路径
    # ------------------------------------------------------------------
    model_path = r"D:\examples_project\python projects\self-enhance-all\ultralytics-main-3\ultralytics-main\runs\obb\train71\weights\best.pt"
    model = YOLO(model_path)

    # ------------------------------------------------------------------
    # 2. 获取模型物理参数 (Params 和 GFLOPS)
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("正在计算模型参数量和计算量...")
    # model.info() 会自动打印 Layers, Parameters, Gradients, GFLOPs99
    model.info(detailed=True)
    print("=" * 50 + "\n")

    # ------------------------------------------------------------------
    # 3. 运行验证 (获取 mAP 和 速度)
    # ------------------------------------------------------------------
    # 注意：为了测出论文里的 'High-Speed mAP'，你的 data_val.yaml
    # 指向的必须是【模糊/高速】测试集。
    results = model.val(
        data="data_val.yaml",
        # data="data_valblur.yaml",
        split="val",
        imgsz=640,
        batch=1,  # ⚠️ 改为 1 以便更准确地测试单帧 FPS (模拟真实流)
        # conf=0.5,  # 常用阈值
        save_json=False,
        save=True,
        task="obb",
        # conf=0.3,
        # iou=0.9,
        iou=0.96,  #模糊训练集0.8
        # iou=0.96, #普通训练集0.93
        workers=0
    )

    # ------------------------------------------------------------------
    # 4. 提取并格式化输出 (直接对应论文表格)
    # ------------------------------------------------------------------
    # 提取 mAP
    map50 = results.box.map50 * 100  # 对应论文 mAP@0.5
    map5095 = results.box.map * 100  # 对应论文 mAP@0.5:0.95

    # 提取速度 (results.speed 是一个字典，单位 ms)
    # 包含 preprocess, inference, postprocess
    speed_dict = results.speed
    inference_time = speed_dict['inference']
    total_time = speed_dict['preprocess'] + speed_dict['inference'] + speed_dict['postprocess']

    # 计算 FPS (1000ms / total_time_per_image)
    fps = 1000.0 / total_time if total_time > 0 else 0

    print("\n" + "#" * 60)
    print("【论文数据直接摘录区】请将以下数据填入你的论文")
    print("#" * 60)

    print(f"{'指标 (Metric)':<25} | {'数值 (Value)':<15} | {'论文对应位置'}")
    print("-" * 65)
    print(f"{'mAP@0.5':<25} | {map50:.1f}%          | 表2 / 摘要")
    print(f"{'mAP@0.5:0.95':<25} | {map5095:.1f}%          | 表2 / 摘要")
    print(f"{'Inference Time':<25} | {inference_time:.1f} ms        | 文本分析")
    print(f"{'Total Latency':<25} | {total_time:.1f} ms        | (用于计算FPS)")
    print(f"{'FPS (Frame/s)':<25} | {fps:.1f}             | 表2 / 摘要")
    print("-" * 65)
    print("注意：Params 和 GFLOPS 请查看上方 model.info() 的输出日志")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()