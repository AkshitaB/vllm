from vllm.model_executor.models.olmo_new import RMSNorm
from olmo.model import *
import torch
from olmo.config import TrainConfig
tc = TrainConfig.load("/home/akshitab/OLMo/configs/peteish-tiny.yaml")

@torch.inference_mode()
def func():
    config = tc.model
    config.init_device = "cuda"

    o_rms = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

    v_rms = RMSNorm(config.d_model, eps=config.layer_norm_eps)

    x = torch.randn(8, 512)

    print(o_rms(x.cuda()))
    print(v_rms.forward_native(x))

func()
