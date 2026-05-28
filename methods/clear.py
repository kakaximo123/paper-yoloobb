import os
import shutil


def clear_directory(target_dir):
    """清空指定文件夹内的所有内容（包含所有文件和子文件夹）."""
    if not os.path.exists(target_dir):
        print(f"路径 {target_dir} 不存在！")
        return

    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)

        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                # 如果是文件或软链接，用 os.remove 删除
                os.remove(item_path)
                print(f"已删除文件: {item_path}")
            elif os.path.isdir(item_path):
                # 如果是文件夹，用 shutil.rmtree 删除
                shutil.rmtree(item_path)
                print(f"已删除文件夹: {item_path}")
        except Exception as e:
            print(f"删除 {item_path} 失败，原因: {e}")


# 使用示例
folder_path = r"D:\examples_project\ultralytics-main\runs\detect"  # 替换为你的真实路径
clear_directory(folder_path)
