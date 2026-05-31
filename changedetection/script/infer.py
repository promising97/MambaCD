import sys

sys.path.append('/data/lgl/hny')
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from MambaCD.changedetection.models.MambaBCD_spatial import STMambaBCD

# -------------------
# 配置参数
# -------------------
pre_image_dir = '/data/lgl/hny/datasets/California_for_GLCD_DA/test/T1'
post_image_dir = '/data/lgl/hny/datasets/California_for_GLCD_DA/test/T2'
output_dir = '/data/lgl/hny/MambaCD/results'
model_path = '/data/lgl/hny/MambaCD/changedetection/saved_models/California_for_GLCD_DA_all/HAM-CD_StarBlock_onlySMB_batch8_all_SAR_iter300000_1778873536.124655/26000_model.pth'

patch_size = 256
in_chans = 3
num_classes = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# -------------------
# 加载模型
# -------------------
model = STMambaBCD(
    pretrained=None,
    patch_size=patch_size,
    in_chans=in_chans,
    num_classes=num_classes,
    depths=[2, 2, 6, 2],
    dims=[64, 128, 320, 512],
    ssm_d_state=64,
    mlp_ratio=4,
    dt_init="random",  # 注意：之前你报过 NotImplementedError，这里最好用字符串 "random" 或模型支持的配置
    use_checkpoint=False
)

# 增加 map_location 保证设备兼容性
state_dict = torch.load(model_path, map_location=device)

# [可选] 兼容 DataParallel 训练出来的权重 (带有 'module.' 前缀)
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()


# -------------------
# 图像预处理函数
# -------------------
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),  # 如果原图不是256，建议改用中心裁剪或滑窗
        transforms.ToTensor(),
        # 👇 极其重要：请去 train 的 dataset 里确认是否有 Normalize。如果有，必须加上！
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


# -------------------
# 遍历文件夹进行推理
# -------------------
pre_images = sorted(os.listdir(pre_image_dir))

for pre_name in pre_images:
    # 只处理图片文件
    if not pre_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        continue

    pre_path = os.path.join(pre_image_dir, pre_name)

    # 【优化】基于 T1 的名字去 T2 文件夹找对应的图，比直接 zip 更稳妥
    post_path = os.path.join(post_image_dir, pre_name)

    if not os.path.exists(post_path):
        print(f"⚠️ 警告: 找不到对应的后时相图像 {post_path}，已跳过。")
        continue

    pre_tensor = preprocess_image(pre_path)
    post_tensor = preprocess_image(post_path)

    with torch.no_grad():
        output = model(pre_tensor, post_tensor)
        output = F.softmax(output, dim=1)
        pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

        # 保存结果为png
    save_path = os.path.join(output_dir, f"{os.path.splitext(pre_name)[0]}_pred.png")
    pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    pred_img.save(save_path)
    print(f"Saved: {save_path}")

print("Inference done!")