import os
import shutil
def delete_folder_if_exists(folder_path):
    """
    如果文件夹存在，则删除该文件夹（包括其中所有文件和子文件夹）
    :param folder_path: 文件夹路径（相对/绝对）
    """
    # 第一步：检查路径是否存在且是文件夹
    if os.path.exists(folder_path):
        if os.path.isdir(folder_path):
            try:
                # 删除文件夹及内部所有内容
                shutil.rmtree(folder_path)
                # print(f"✅ 文件夹已成功删除：{folder_path}")
            except Exception as e:
                # 捕获删除失败的异常（如权限不足、文件被占用等）
                print(f"❌ 删除文件夹失败：{folder_path}，错误信息：{str(e)}")
        else:
            # 路径存在但不是文件夹（是文件）
            print(f"⚠️ 路径存在但不是文件夹，无法删除：{folder_path}")
    else:
        # 路径不存在
        print(f"ℹ️ 文件夹不存在，无需删除：{folder_path}")