import os


def write_list_from_folder(image_dir, output_file, remove_extension=True):
    """
    读取指定文件夹下的所有图片，并将其名称写入目标 txt 文件中。
    """
    # 支持的图片格式
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

    if not os.path.exists(image_dir):
        print(f"❌ 错误：找不到文件夹 {image_dir}，请检查路径是否正确。")
        return 0

    image_names = []
    # 使用 sorted 进行排序，让列表看起来更整洁
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(valid_extensions):
            if remove_extension:
                # 只保留文件名，去掉后缀
                name_only = os.path.splitext(filename)[0]
                image_names.append(name_only)
            else:
                # 保留完整文件名 (含后缀)
                image_names.append(filename)

    if not image_names:
        print(f"⚠️ 警告：在 {image_dir} 中没有找到支持的图片文件。")
        return 0

    # 确保输出 txt 文件的父级目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 写入 txt 文件
    with open(output_file, 'w') as f:
        for name in image_names:
            f.write(f"{name}\n")

    print(f"✅ 成功！从 {image_dir} 读取了 {len(image_names)} 张图片 -> 已保存至 {output_file}")
    return len(image_names)


if __name__ == '__main__':
    # ================= 配置区域 =================
    # 1. 你的 train 和 test 图片文件夹绝对路径
    TRAIN_FOLDER = '/data/lgl/hny/datasets/new_small/train/T1'
    TEST_FOLDER = '/data/lgl/hny/datasets/new_small/test/T1'

    # 2. txt 文件保存的目录路径
    OUTPUT_FOLDER = '/data/lgl/hny/datasets/new_small'

    # 3. 如果你的网络模型需要带后缀的名字（如 image.png），把这里改成 False
    REMOVE_EXT = False
    # ============================================

    train_txt_path = os.path.join(OUTPUT_FOLDER, 'train.txt')
    test_txt_path = os.path.join(OUTPUT_FOLDER, 'test.txt')

    print("开始生成列表...")
    print("-" * 40)

    # 生成 train.txt
    write_list_from_folder(
        image_dir=TRAIN_FOLDER,
        output_file=train_txt_path,
        remove_extension=REMOVE_EXT
    )

    # 生成 test.txt
    write_list_from_folder(
        image_dir=TEST_FOLDER,
        output_file=test_txt_path,
        remove_extension=REMOVE_EXT
    )

    print("-" * 40)
    print("全部执行完毕！")