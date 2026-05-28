import os


def convert_labels_to_single_class(input_folder, output_folder, target_class_id="0"):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出目录: {output_folder}")

    # 获取文件夹中所有的 txt 文件
    files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

    if not files:
        print("未在输入文件夹中找到 .txt 文件。")
        return

    print(f"开始处理，共计 {len(files)} 个文件...")

    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                # 将第一列（类别ID）修改为指定的 target_class_id
                parts[0] = target_class_id
                # 重新组合成一行，保持后面的坐标不变
                new_lines.append(" ".join(parts) + "\n")

        # 写入新文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    print(f"处理完成！所有修改后的文件已保存至: {output_folder}")


# --- 配置参数 ---
input_dir = r"D:\examples_project\dataset\train_data\labels\val"  # 替换为你原始txt文件的路径
output_dir = r"D:\examples_project\dataset\train_data\labels\val1"  # 替换为你想要保存结果的路径

convert_labels_to_single_class(input_dir, output_dir)
