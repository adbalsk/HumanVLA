
from .HITR_carry            import HITRCarryEnv
from .HITR_rearrangement    import HITRRearrangementEnv
from .HITR_VLA              import HITRRearrangementVLAEnv
#from .sit                   import SitEnv
#from .sit_simple            import SitSimpleEnv
from .sit_obs               import SitEnv
from .sit_vision            import SitVisionEnv
from .sit_vision_test       import SitVisionTestEnv

def build_env(cfg):
    return eval(cfg.name)(cfg)