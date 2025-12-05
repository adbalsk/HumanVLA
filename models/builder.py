from .amp.amp_trainer import AMPTrainer
from .amp.amp_player import AMPPlayer
from .dagger.dagger_trainer import DaggerTrainer
from .dagger.dagger_player  import DaggerPlayer
from .dagger_test.dagger_player import DaggerTestPlayer
from .dagger_test.dagger_trainer import DaggerTestTrainer
from .dagger_rl.dagger_player import DaggerRLPlayer
from .dagger_rl.dagger_trainer import DaggerRLTrainer
from .amp_stage2.amp_trainer import AMPStage2Trainer
from .amp_stage2.amp_player import AMPStage2Player

def build_model(cfg, env):
    model_name = cfg.model
    if cfg.test:
        model_name = model_name + 'Player'
    else:
        model_name = model_name + 'Trainer'
    return eval(model_name)(cfg, env)