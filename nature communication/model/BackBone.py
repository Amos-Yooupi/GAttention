from GAttention import GAttention
import torch.nn as nn


class BackBone(nn.Module):
    def __init__(self, config, backbone_type):
        super(BackBone, self).__init__()

        self.name = backbone_type
        if backbone_type == 'GAttention':
            self.backbone = GAttention(config.embed_dim, config.hidden_dim,
                                       config.in_len, config.num_head_for_time,
                                       config.num_head_for_node, config.num_moe_layer,
                                       config.num_expert, config.top_k, config.att_mode,
                                       config.activation, config.is_fusion, config.num_blocks)
        else:
            raise ValueError('Invalid backbone type: {}'.format(backbone_type))

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)
