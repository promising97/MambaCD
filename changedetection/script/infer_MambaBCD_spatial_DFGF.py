import sys

sys.path.append("/root/autodl-tmp/MambaCD")

import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader

from MambaCD.changedetection.configs.spatialmamba.config import get_config
from MambaCD.changedetection.datasets.make_data_loader import ChangeDetectionDatset
from MambaCD.changedetection.models.ChangeDecoder_Star_GaborFPU_DFGF import (
    ChangeDecoder,
)
from MambaCD.changedetection.models.MambaBCD_spatial_DFGF import STMambaBCD
from MambaCD.changedetection.utils_func.metrics import Evaluator


MODEL_STACK = "SpatialMamba + ChangeDecoder_Star_GaborFPU_DFGF"


def extract_state_dict(checkpoint):
    """Extract and normalize a model state dict from common checkpoint formats."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}")

    for key in ("model", "state_dict", "model_state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break

    state_dict = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("module."):
            key = key[len("module.") :]
        state_dict[key] = value

    if not state_dict:
        raise RuntimeError("No tensor parameters were found in the checkpoint")
    return state_dict


class Inference:
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        if args.batch_size != 1:
            raise ValueError(
                "This script saves one named prediction at a time; use --batch_size 1"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            # These three switches must match the DFGF training experiment.
            use_dfgf=True,
            dfgf_use_frequency_gate=True,
            dfgf_use_direction_selector=True,
        )

        if not isinstance(self.deep_model.decoder, ChangeDecoder):
            raise RuntimeError(
                "Decoder mismatch: expected ChangeDecoder_Star_GaborFPU_DFGF, "
                f"got {type(self.deep_model.decoder).__name__}"
            )

        print(f"[Model] stack={MODEL_STACK}")
        print("[Model] use_dfgf=True, frequency_gate=True, direction_selector=True")

        self.deep_model = self.deep_model.to(self.device)

        self.change_map_saved_path = os.path.join(
            args.result_saved_path,
            args.dataset,
            args.model_type,
            "change_map",
        )
        os.makedirs(self.change_map_saved_path, exist_ok=True)

        self._load_checkpoint(args.resume)
        self.deep_model.eval()

    def _load_checkpoint(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)

        # Strict loading is intentional: inference must never continue with a
        # partially initialized model caused by using the wrong architecture.
        try:
            self.deep_model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "The checkpoint does not exactly match the DFGF model. Verify that "
                "MambaBCD_spatial_DFGF.py, ChangeDecoder_Star_GaborFPU_DFGF.py, "
                "StarIR_gabor_DFGF.py, the YAML config, and all DFGF switches are "
                "the same versions used for training.\n"
                f"Original error:\n{exc}"
            ) from exc

        print(f"[Checkpoint] loaded strictly from: {checkpoint_path}")

    def infer(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

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
            num_workers=self.args.num_workers,
            drop_last=False,
            pin_memory=self.device.type == "cuda",
        )

        print(f"[Data] samples={len(dataset)}")
        print(f"[Data] test_path={self.args.test_dataset_path}")
        print(f"[Output] {self.change_map_saved_path}")

        self.evaluator.reset()

        with torch.no_grad():
            for data in val_data_loader:
                pre_change_imgs, post_change_imgs, labels, names = data

                pre_change_imgs = pre_change_imgs.to(
                    self.device, dtype=torch.float32, non_blocking=True
                )
                post_change_imgs = post_change_imgs.to(
                    self.device, dtype=torch.float32, non_blocking=True
                )
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)

                logits = self.deep_model(pre_change_imgs, post_change_imgs)
                predictions = torch.argmax(logits, dim=1)

                pred_np = predictions.cpu().numpy()
                label_np = labels.cpu().numpy()
                self.evaluator.add_batch(label_np, pred_np)

                original_name = os.path.basename(str(names[0]))
                image_name = os.path.splitext(original_name)[0] + ".png"

                pred = np.squeeze(pred_np)
                gt = np.squeeze(label_np)

                # Error-analysis map:
                # TP white, FP red, FN green, TN black.
                color_map = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                color_map[(pred == 1) & (gt == 1)] = [255, 255, 255]
                color_map[(pred == 1) & (gt == 0)] = [255, 0, 0]
                color_map[(pred == 0) & (gt == 1)] = [0, 255, 0]
                color_map[(pred == 0) & (gt == 0)] = [0, 0, 0]

                imageio.imwrite(
                    os.path.join(self.change_map_saved_path, image_name), color_map
                )

        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        recall = self.evaluator.Pixel_Recall_Rate()
        precision = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kappa = self.evaluator.Kappa_coefficient()

        print(
            f"Recall={recall}, Precision={precision}, OA={oa}, "
            f"F1={f1_score}, IoU={iou}, Kappa={kappa}"
        )
        print("Inference stage is done!")


def main():
    parser = argparse.ArgumentParser(
        description="Inference for SpatialMamba + Star-FPU-DFGF change detection"
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
        default=None,
        nargs="+",
        help="Override YAML options with KEY VALUE pairs",
    )
    parser.add_argument(
        "--pretrained_weight_path",
        type=str,
        default=(
            "/root/autodl-tmp/MambaCD/pretrained_weight/"
            "spatialmamba_small_224_1k.pth"
        ),
    )
    parser.add_argument("--dataset", type=str, default="California_for_GLCD_DA_all")
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="/root/autodl-tmp/California_for_GLCD_DA/test",
    )
    parser.add_argument(
        "--test_data_list_path",
        type=str,
        default="/root/autodl-tmp/California_for_GLCD_DA/test_list.txt",
    )
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--model_type",
        type=str,
        default="HAM-CD_Star_FPU_DFGF_36000",
    )
    parser.add_argument(
        "--result_saved_path",
        type=str,
        default="/root/autodl-tmp/MambaCD/results",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=(
            "/root/autodl-tmp/MambaCD/changedetection/saved_models/"
            "California_for_GLCD_DA_all/"
            "HAM-CD_Star_FPU_DFGF_1785231530.3483572/36000_model.pth"
        ),
    )

    args = parser.parse_args()

    if not os.path.isfile(args.test_data_list_path):
        raise FileNotFoundError(f"Test list not found: {args.test_data_list_path}")

    with open(args.test_data_list_path, "r", encoding="utf-8") as file:
        args.test_data_name_list = [
            line.strip() for line in file if line.strip()
        ]

    if not args.test_data_name_list:
        raise RuntimeError(f"Test list is empty: {args.test_data_list_path}")

    inference = Inference(args)
    inference.infer()


if __name__ == "__main__":
    main()
