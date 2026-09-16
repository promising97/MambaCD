"""Train MambaBCD-Spatial with DFGF-Star and the COPE-S branch."""

import argparse
import os
import sys
import time


PROJECT_ROOT = "/root/autodl-tmp/MambaCD"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# "0" means the first physical GPU. Use "1" for the second physical GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from MambaCD.changedetection.configs.spatialmamba.config import get_config
from MambaCD.changedetection.datasets.make_data_loader import (
    ChangeDetectionDatset,
    make_data_loader,
)
from MambaCD.changedetection.models.MambaBCD_spatial_COPE_S_DFGF import STMambaBCD
from MambaCD.changedetection.utils_func.edge_loss import EdgeLoss
from MambaCD.changedetection.utils_func.metrics import Evaluator
import MambaCD.changedetection.utils_func.lovasz_loss as L


class Tee:
    """Write stdout/stderr to both terminal and log file."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


def setup_log_file(log_path):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    log_file = open(
        log_path,
        "a",
        encoding="utf-8",
        buffering=1,
    )
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"Logging to {log_path}")


def load_compatible_checkpoint(model, checkpoint_path):
    """Load matching tensors from an old or new full-model checkpoint.

    This permits initialization of the DFGF model from an older Gabor model
    while skipping missing or shape-incompatible tensors.
    """
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError(
            f"=> no checkpoint found at '{checkpoint_path}'"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a state_dict or a dictionary containing "
            "'state_dict'/'model'."
        )

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "model" in checkpoint:
        checkpoint = checkpoint["model"]

    checkpoint = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if torch.is_tensor(value)
    }

    current_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in checkpoint.items()
        if key in current_state
        and value.shape == current_state[key].shape
    }
    skipped_keys = [
        key
        for key, value in checkpoint.items()
        if key not in current_state
        or value.shape != current_state[key].shape
    ]

    load_result = model.load_state_dict(
        compatible_state,
        strict=False,
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Compatible tensors: {len(compatible_state)}")
    print(f"Skipped tensors: {len(skipped_keys)}")
    print(f"Missing tensors: {len(load_result.missing_keys)}")
    print(f"Unexpected tensors: {len(load_result.unexpected_keys)}")

    if skipped_keys:
        print("First skipped keys:")
        for key in skipped_keys[:20]:
            print(f"  - {key}")

    return load_result


class Trainer:
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)
        self.evaluator = Evaluator(num_class=2)

        self.deep_model = STMambaBCD(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.SPATIALMAMBA.PATCH_SIZE,
            in_chans=config.MODEL.SPATIALMAMBA.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.SPATIALMAMBA.DEPTHS,
            dims=config.MODEL.SPATIALMAMBA.EMBED_DIM,
            ssm_d_state=config.MODEL.SPATIALMAMBA.D_STATE,
            mlp_ratio=config.MODEL.SPATIALMAMBA.MLP_RATIO,
            dt_init=config.MODEL.SPATIALMAMBA.DT_INIT,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_dfgf=args.use_dfgf,
            dfgf_use_frequency_gate=args.dfgf_frequency_gate,
            dfgf_use_direction_selector=args.dfgf_direction_selector,
            use_cope_s=args.use_cope_s,
            cope_use_change_guide=args.cope_change_guide,
            cope_use_direction_reweight=args.cope_direction_reweight,
            cope_use_residual=args.cope_residual,
        )

        print(
            "DFGF configuration: "
            f"use_dfgf={args.use_dfgf}, "
            f"frequency_gate={args.dfgf_frequency_gate}, "
            f"direction_selector={args.dfgf_direction_selector}"
        )
        print(
            "SpatialMamba pretrained weight: "
            f"{args.pretrained_weight_path}"
        )
        print(
            "COPE-S configuration: "
            f"use_cope_s={args.use_cope_s}, "
            f"change_guide={args.cope_change_guide}, "
            f"direction_reweight={args.cope_direction_reweight}, "
            f"residual={args.cope_residual}"
        )

        if not args.cuda:
            raise RuntimeError(
                "This training script is configured for CUDA training. "
                "Pass --cuda or remove this check for CPU training."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        self.deep_model = self.deep_model.cuda()

        self.model_save_path = os.path.join(
            args.model_param_path,
            args.dataset,
            f"{args.model_type}_{time.time()}",
        )
        os.makedirs(self.model_save_path, exist_ok=True)

        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        # --pretrained-weight-path initializes only the SpatialMamba encoder.
        # --resume loads a full change-detection model checkpoint.
        if args.resume is not None:
            load_compatible_checkpoint(
                self.deep_model,
                args.resume,
            )

        self.optim = optim.AdamW(
            self.deep_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.edge_loss = EdgeLoss()

    def training(self):
        best_f1 = 0.0
        best_round = []

        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)

        for _ in tqdm(range(elem_num)):
            itera, data = next(train_enumerator)
            pre_change_imgs, post_change_imgs, labels, _ = data

            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda().float()
            labels = labels.cuda().long()

            output_1 = self.deep_model(
                pre_change_imgs,
                post_change_imgs,
            )

            self.optim.zero_grad()

            ce_loss_1 = F.cross_entropy(
                output_1,
                labels,
                ignore_index=255,
            )
            lovasz_loss = L.lovasz_softmax(
                F.softmax(output_1, dim=1),
                labels,
                ignore=255,
            )

            # Keep the loss definition identical to the supplied script.
            main_loss = ce_loss_1 + 2.0 * lovasz_loss
            final_loss = main_loss

            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.deep_model.parameters(),
                max_norm=1.0,
            )
            self.optim.step()

            if (itera + 1) % 10 == 0:
                print(
                    f"iter is {itera + 1}, "
                    f"ce is {ce_loss_1.item():.4f}, "
                    f"lovasz is {lovasz_loss.item():.4f}, "
                    f"overall loss is {final_loss.item():.4f}"
                )

            if (itera + 1) % 2000 == 0:
                self.deep_model.eval()
                rec, pre, oa, f1_score, iou, kc = self.validation()

                if f1_score > best_f1:
                    checkpoint_path = os.path.join(
                        self.model_save_path,
                        f"{itera + 1}_model.pth",
                    )
                    torch.save(
                        self.deep_model.state_dict(),
                        checkpoint_path,
                    )
                    best_f1 = f1_score
                    best_round = [
                        rec,
                        pre,
                        oa,
                        f1_score,
                        iou,
                        kc,
                    ]
                    print(f"Saved best model to {checkpoint_path}")

                self.deep_model.train()

        print("The accuracy of the best round is", best_round)

    def validation(self):
        print("---------starting evaluation-----------")
        self.evaluator.reset()

        dataset = ChangeDetectionDatset(
            self.args.test_dataset_path,
            self.args.test_data_name_list,
            self.args.crop_size,
            None,
            "test",
        )
        val_data_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            drop_last=False,
        )

        torch.cuda.empty_cache()

        with torch.no_grad():
            for data in val_data_loader:
                pre_change_imgs, post_change_imgs, labels, _ = data

                pre_change_imgs = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda().float()
                labels = labels.cuda().long()

                output_1 = self.deep_model(
                    pre_change_imgs,
                    post_change_imgs,
                )

                prediction = output_1.detach().cpu().numpy()
                prediction = np.argmax(prediction, axis=1)
                labels = labels.cpu().numpy()

                self.evaluator.add_batch(labels, prediction)

        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()

        print(
            f"Recall rate is {rec}, "
            f"Precision rate is {pre}, "
            f"OA is {oa}, "
            f"F1 score is {f1_score}, "
            f"IoU is {iou}, "
            f"Kappa coefficient is {kc}"
        )

        return rec, pre, oa, f1_score, iou, kc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training on SYSU/LEVIR-CD+/WHU-CD dataset"
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default=(
            "/root/autodl-tmp/MambaCD/changedetection/configs/"
            "spatialmamba/spatialmamba_small.yaml"
        ),
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--pretrained_weight_path",
        type=str,
        default=(
            "/root/autodl-tmp/MambaCD/pretrained_weight/"
            "spatialmamba_small_224_1k.pth"
        ),
    )

    # DFGF ablation switches. Python 3.12 supports BooleanOptionalAction:
    # --use-dfgf / --no-use-dfgf, etc.
    parser.add_argument(
        "--use-dfgf",
        dest="use_dfgf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the direction-frequency gated Gabor module.",
    )
    parser.add_argument(
        "--dfgf-frequency-gate",
        dest="dfgf_frequency_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable learnable frequency-domain gating.",
    )
    parser.add_argument(
        "--dfgf-direction-selector",
        dest="dfgf_direction_selector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable adaptive Gabor orientation selection.",
    )

    # COPE-S ablation switches. The original GaborFPU stack remains in the old
    # files; these switches isolate the components of the new third innovation.
    parser.add_argument(
        "--use-cope-s",
        dest="use_cope_s",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the COPE-S branch; disable for branch-removal ablation.",
    )
    parser.add_argument(
        "--cope-change-guide",
        dest="cope_change_guide",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the bi-temporal spatial change guide (CGG).",
    )
    parser.add_argument(
        "--cope-direction-reweight",
        dest="cope_direction_reweight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable change-conditioned direction reweighting (DRS).",
    )
    parser.add_argument(
        "--cope-residual",
        dest="cope_residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable residual channel fusion (RCF).",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="California",
    )
    parser.add_argument("--type", type=str, default="train")
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="/root/autodl-tmp/data/California/train",
    )
    parser.add_argument(
        "--train_data_list_path",
        type=str,
        default="/root/autodl-tmp/data/California/train_list.txt",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="/root/autodl-tmp/data/California/test",
    )
    parser.add_argument(
        "--test_data_list_path",
        type=str,
        default="/root/autodl-tmp/data/California/test_list.txt",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--train_data_name_list", type=list)
    parser.add_argument("--test_data_name_list", type=list)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max_iters", type=int, default=300000)
    parser.add_argument(
        "--model_type",
        type=str,
        default="LEM-CD_COPE-S_DFGF_yiyuan_batch8_all_SAR_iter300000",
    )
    parser.add_argument(
        "--model_param_path",
        type=str,
        default=(
            "/root/autodl-tmp/MambaCD/changedetection/saved_models"
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Full change-detection model checkpoint, not encoder pretraining.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Save training stdout/stderr to this log file.",
    )

    return parser.parse_args()


def load_dataset_lists(args):
    with open(
        args.train_data_list_path,
        "r",
        encoding="utf-8",
    ) as file:
        args.train_data_name_list = [
            data_name.strip()
            for data_name in file
            if data_name.strip()
        ]

    with open(
        args.test_data_list_path,
        "r",
        encoding="utf-8",
    ) as file:
        args.test_data_name_list = [
            data_name.strip()
            for data_name in file
            if data_name.strip()
        ]


def main():
    args = parse_args()

    if args.log_path:
        setup_log_file(args.log_path)

    load_dataset_lists(args)

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
