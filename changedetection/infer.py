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
model_path = '/data/lgl/hny/MambaCD/saved_models/SYSU/MambaBCD_1778723884.0825462/14000_model.pth'

patch_size = 256  # 与训练时一致
in_chans = 3      # 输入通道数，根据你的模型调整
num_classes = 2   # 二分类
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
    depths=[2, 2, 6, 2],   # 根据训练配置
    dims=[64, 128, 320, 512],
    ssm_d_state=64,         # 根据训练配置
    mlp_ratio=4,
    dt_init=0.001,
    use_checkpoint=False
)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# -------------------
# 图像预处理函数
# -------------------
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, C, H, W]
    return img_tensor

# -------------------
# 遍历文件夹进行推理
# -------------------
pre_images = sorted(os.listdir(pre_image_dir))
post_images = sorted(os.listdir(post_image_dir))

for pre_name, post_name in zip(pre_images, post_images):
    pre_path = os.path.join(pre_image_dir, pre_name)
    post_path = os.path.join(post_image_dir, post_name)

    pre_tensor = preprocess_image(pre_path)
    post_tensor = preprocess_image(post_path)

    with torch.no_grad():
        output = model(pre_tensor, post_tensor)  # [B, C, H, W]
        output = F.softmax(output, dim=1)
        pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]  # [H, W]

    # 保存结果为png
    save_path = os.path.join(output_dir, f"{os.path.splitext(pre_name)[0]}_pred.png")
    pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    pred_img.save(save_path)
    print(f"Saved: {save_path}")

print("Inference done!")