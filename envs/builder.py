
from .HITR_carry            import HITRCarryEnv
from .HITR_rearrangement    import HITRRearrangementEnv
from .HITR_VLA              import HITRRearrangementVLAEnv
from .sit                   import SitEnv

def build_env(cfg):
    return eval(cfg.name)(cfg)