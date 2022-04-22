from curses import meta
import json
from overrides import overrides
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import Visualizer

from pano_seg.mask2former import add_maskformer2_config
from pano_seg.predictors.predictor_base import PredictorBase
from pano_seg.mask2former.maskformer_model import MaskFormer


class MaskFormerWrapper(MaskFormer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _preprocess_input(self, batched_inputs) -> ImageList:
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        return ImageList.from_tensors(images, self.size_divisibility)

    def forward(self, batched_inputs):
        """Overrides the forward method of the parent class
        so that
        """
        # Preprocess input
        images = self._preprocess_input(batched_inputs)

        # Forward pass
        # features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(self.backbone(images.tensor))

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results = []

        for mask_cls_result, mask_pred_result, _, _ in zip(
            mask_cls_results,
            mask_pred_results,
            batched_inputs,
            images.image_sizes,
        ):
            panoptic_seg, segments_info, mask_logits, mask_probs = retry_if_cuda_oom(
                self.panoptic_inference
            )(mask_cls_result, mask_pred_result)
            processed_results.append(
                {
                    "panoptic_seg": panoptic_seg,
                    "segments_info": segments_info,
                    "mask_logits": mask_logits,
                    "mask_probs": mask_probs,
                }
            )

        return processed_results

    def panoptic_inference(self, mask_cls, mask_logits):
        """Overrides the panoptic_inference method of the parent class
        so that also the raw mask logits and the mask probs tensors are
        returned as result.
        """
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_logits.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (
            scores > self.object_mask_threshold
        )
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            mask_max_probs, cur_mask_ids = cur_prob_masks.max(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = (
                    pred_class
                    in self.metadata.thing_dataset_id_to_contiguous_id.values()
                )
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info, mask_logits, mask_max_probs


def _make_mask2former_cfg(config_file_path: Path, checkpoint_file_path: Path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(config_file_path))
    cfg.MODEL.WEIGHTS = str(checkpoint_file_path)

    return cfg


class Mask2FormerPredictor(PredictorBase):
    """Adapted from detectron2 DefaultPredictor"""

    def __init__(self, model_dir_path: Path, visualize: bool = True):

        self.visualize = visualize

        if not model_dir_path.is_dir():
            raise FileNotFoundError(f"{model_dir_path} is not a valid directory!")
        config_file_path = model_dir_path / "config.yaml"

        if not config_file_path.is_file():
            raise FileNotFoundError("Model config not found!")
        checkpoint_file_path = model_dir_path / "model_final.pth"
        if not checkpoint_file_path.is_file():
            raise FileNotFoundError("Model checkpoint not found!")

        self.cfg = _make_mask2former_cfg(config_file_path, checkpoint_file_path)

        self.model = MaskFormerWrapper(self.cfg)
        self.model.to(torch.device(self.cfg.MODEL.DEVICE))
        self.model.eval()

        metadata_file_path = model_dir_path / "metadata.json"
        if not metadata_file_path.is_file():
            raise FileNotFoundError("Model metadata file not found!")
        with metadata_file_path.open("r") as f:
            metadata_dict = json.load(f)

        # Hack to avoid error due to different dataset name
        del metadata_dict["name"]
        self.model.metadata.set(**metadata_dict)

        # For reverse lookup of category id
        self.contiguous_id_to_dataset_id = {
            int(v): int(k)
            for k, v in self.model.metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        checkpointer = DetectionCheckpointer(self.model)
        _ = checkpointer.load(self.cfg.MODEL.WEIGHTS)

        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    @overrides
    def __call__(self, original_image: np.ndarray):
        with torch.no_grad():
            image = original_image.copy()
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]

            if self.visualize:
                visualizer = Visualizer(original_image, metadata=self.model.metadata)
                vis_image = visualizer.draw_panoptic_seg(
                    predictions["panoptic_seg"].cpu(), predictions["segments_info"]
                )
                predictions["panoptic_seg_vis"] = vis_image.get_image()

            for k in ["panoptic_seg", "mask_logits", "mask_probs"]:
                predictions[k] = predictions[k].cpu().numpy()

            # Category id reverse lookup
            for sinfo in predictions["segments_info"]:
                sinfo["category_id"] = self.contiguous_id_to_dataset_id[sinfo["category_id"]]

            return predictions
