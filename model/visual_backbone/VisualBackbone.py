import math
from detectron2.modeling import META_ARCH_REGISTRY
import torch
import torch.utils.checkpoint
from torch import nn
from model.visual_backbone.convert_sync_bacchnorm import my_convert_sync_batchnorm
import detectron2
from .detectron2_config import add_layoutlmv2_config
class VisualBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = detectron2.config.get_cfg()
        add_layoutlmv2_config(self.cfg)
        meta_arch = self.cfg.MODEL.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(self.cfg)
        assert isinstance(model.backbone, detectron2.modeling.backbone.FPN)
        self.backbone = model.backbone
        if (
            config.convert_sync_batchnorm
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() > -1
        ):
            self_rank = torch.distributed.get_rank()
            node_size = torch.cuda.device_count()
            world_size = torch.distributed.get_world_size()
            assert world_size % node_size == 0

            node_global_ranks = [
                list(range(i * node_size, (i + 1) * node_size)) for i in range(world_size // node_size)
            ]
            sync_bn_groups = [
                torch.distributed.new_group(ranks=node_global_ranks[i]) for i in range(world_size // node_size)
            ]
            node_rank = self_rank // node_size
            assert self_rank in node_global_ranks[node_rank]

            self.backbone = my_convert_sync_batchnorm(self.backbone, process_group=sync_bn_groups[node_rank])

        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        self.register_buffer(
            "pixel_mean",
            torch.Tensor(self.cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1),
        )
        self.register_buffer("pixel_std", torch.Tensor(self.cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1))
        self.out_feature_key = "p2"
        if torch.is_deterministic():

            input_shape = (224, 224)
            backbone_stride = self.backbone.output_shape()[self.out_feature_key].stride
            self.pool = nn.AvgPool2d(
                (
                    math.ceil(math.ceil(input_shape[0] / backbone_stride) / config.image_feature_pool_shape[0]),
                    math.ceil(math.ceil(input_shape[1] / backbone_stride) / config.image_feature_pool_shape[1]),
                )
            )
        else:
            self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(self.backbone.output_shape()[self.out_feature_key].channels)
        assert self.backbone.output_shape()[self.out_feature_key].channels == config.image_feature_pool_shape[2]

    def forward(self, images):
        # images_input = (images.tensor - self.pixel_mean) / self.pixel_std
        images_input = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = features[self.out_feature_key]
        features = self.pool(features).flatten(start_dim=2).transpose(1, 2).contiguous()
        return features