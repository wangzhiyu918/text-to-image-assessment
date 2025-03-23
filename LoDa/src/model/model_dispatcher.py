import timm

from .loda import LoDa


def create_model(cfg):
    if cfg.model.model_name == "loda":
        basic_model = timm.create_model(
            cfg.model.basic_model_name,
            # img_size=cfg.model.vit_param.img_size,
            pretrained=cfg.model.basic_model_pretrained,
            num_classes=0,
        )
        for param in basic_model.parameters():
            param.requires_grad = False

        net_arch = LoDa(basic_model, cfg=cfg, basic_state_dict=None)
        net_arch.freeze()
    else:
        raise Exception("%s model not supported" % cfg.model.model_name)

    return net_arch
