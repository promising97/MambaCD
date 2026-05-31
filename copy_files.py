import os
import shutil

def copy_images_from_txt(txt_file_path, source_folder, target_folder):
    """
    从txt文件中读取图片列表，并将这些图片从源文件夹复制到目标文件夹
    
    参数:
        txt_file_path (str): 包含图片列表的txt文件路径
        source_folder (str): 图片所在的源文件夹路径
        target_folder (str): 目标文件夹路径
    """
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)
    
    # 读取txt文件中的图片名
    with open(txt_file_path, 'r') as file:
        image_names = [line.strip() for line in file.readlines()]
    
    # 复制图片
    copied_count = 0
    for img_name in image_names:
        src_path = os.path.join(source_folder, img_name)
        dst_path = os.path.join(target_folder, img_name)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)  # 复制文件
            copied_count += 1
        else:
            print(f"Warning: Image {img_name} does not exist in source folder.")
    
    print(f"Completed. Total {copied_count} images copied.")

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际路径
    txt_file = "/data/lgl/datasets/SYSU-CD/test_list.txt"  # 包含图片列表的txt文件
    src_dir = "/data/lgl/datasets/SYSU-CD/test/GT/"          # 图片所在的源文件夹
    dst_dir = "/data/lgl/datasets/SYSU-CD/train/GT/"         # 目标文件夹
    
    copy_images_from_txt(txt_file, src_dir, dst_dir)