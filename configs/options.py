# import sys
# sys.path.append('/')

import argparse
import inspect
import glob
from os import path as osp
import datetime

from omegaconf import OmegaConf
from ruamel.yaml import YAML

from pytorch_lightning.trainer import Trainer

from DLproj.utils.registry import CONFIG_REGISTRY
from DLproj.utils.misc import get_gpu_ids_from_flag
from DLproj.options.parsing import get_args_from_cls_or_func, add_argparse_args
from DLproj.options.yaml_options import convert_to_commented_map, get_yaml_args_types_comments
from DLproj import _GLOBAL_VARS


@CONFIG_REGISTRY.register()
class SDOptions():
    """
    get options for SD training
    行为：
        1、获取 yaml 表参数
        2、将参数暴露给 CLI
    """
    def __init__(self, yaml_path:str):
        self.yaml_path = yaml_path
        self.opt = OmegaConf.load(yaml_path)
        self.update_options()

    def add_argparse_args(self):
        self.args_type, self.args_comment = get_yaml_args_types_comments(yaml_path)
        self.parser = argparse.ArgumentParser()

        self.parser = add_argparse_args(self.parser, self.opt, self.args_type, self.args_comment)
        self._opt = self.parser.parse_args()

    def update_options(self):
        now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

        if self.opt.name and self.opt.resume:
            raise ValueError(
                "-n/--name and -r/--resume cannot be specified both."
                "If you want to resume training in a new log folder, "
                "use -n/--name in combination with --resume_from_checkpoint"
            )
        if self.opt.resume:
            if not osp.exists(self.opt.resume):
                raise ValueError(f"Cannot find {self.opt.resume}")
            if osp.isfile(self.opt.resume):
                paths = self.opt.resume.split("/")
                # idx = len(paths)-paths[::-1].index("logs")+1
                # logdir = "/".join(paths[:idx])
                logdir = "/".join(paths[:-2])
                ckpt = self.opt.resume
            else:
                assert osp.isdir(self.opt.resume), self.opt.resume
                logdir = self.opt.resume.rstrip("/")
                ckpt = osp.join(logdir, "checkpoints", "last.ckpt")

            self.opt.resume_from_checkpoint = ckpt
            # base_configs = sorted(glob.glob(osp.join(logdir, "configs/*.yaml")))
            # self.opt.base = base_configs + self.opt.base
            _tmp = logdir.split("/")
            nowname = _tmp[-1]
        else:
            if self.opt.name:
                name = "_" + self.opt.name
            else:
                cfg_fname = osp.split(self.yaml_path)[-1]
                cfg_name = osp.splitext(cfg_fname)[0]
                name = "_" + cfg_name
            nowname = now + name + self.opt.postfix
            logdir = osp.join(self.opt.logdir, nowname)

        self.opt.logdir = logdir
        self.opt.ckptdir = osp.join(logdir, "checkpoints")
        self.opt.cfgdir = osp.join(logdir, "configs")
        self.opt.now = now
        self.opt.nowname = nowname

        lightning_config = self.opt.pop("lightning", OmegaConf.create())
        trainer_config = self.opt.pop("Trainer", OmegaConf.create())

        # 判断是否使用 gpu，以及 gpu 数量
        trainer_config.gpus = get_gpu_ids_from_flag(trainer_config.gpus)
        self.ngpu = len(trainer_config.gpus)
        if self.ngpu > 0:
            print(f"Running on GPUs {trainer_config.gpus}")
        else:
            print("Running on CPU")

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": self.opt.debug,
                    "id": nowname,
                }
            },
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": "tensorboard",
                    "save_dir": logdir,
                }
            },
        }
        self.logger_config = OmegaConf.merge(default_logger_cfgs["tensorboard"], lightning_config.get("logger", OmegaConf.create()))

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        self.default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": self.opt.ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        # add callback which sets up log directory
        self.default_callbacks_cfg = {
            "setup_callback": {
                "target": "ldm.callbacks.SetupCallback",
                "params": {
                    "resume": self.opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": self.opt.ckptdir,
                    "cfgdir": self.opt.cfgdir,
                    "config": self.opt,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "ldm.callbacks.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "ldm.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "ldm.callbacks.CUDACallback"
            },
        }

        lightning_config.trainer = trainer_config
        self.lightning_config = lightning_config




class SDOptionsGenerator():
    """
    Generate options for SD training
    基本流程：
        1、获取模组参数，如模型类、数据类等
        2、获取基本参数，如环境参数等
        3、从额外的 yaml 表中更新参数

        支持将参数导出到一张 yaml 总表中

        另外也支持将参数暴露给 argparse，流程上 argparse 的参数会覆盖 yaml 表中的参数，这样可以在命令行中直接修改参数（这个可以直接写到总类中）
        而如果是在 IDE 中运行，可以直接修改 yaml 表中的参数或者直接修改 self.opt 中的参数
    """
    def __init__(self):
        self.opt = {}
        self.type_info = {}
        self.docs = {}

        trainer_name, trainer_default, trainer_type, trainer_doc = get_args_from_cls_or_func(Trainer)
        trainer_module_name = trainer_name.split(".")[-1]
        self.opt[trainer_module_name] = trainer_default
        # self.opt[trainer_module_name]["module_path"] = trainer_name
        self.type_info[trainer_module_name] = trainer_type
        self.docs[trainer_module_name] = trainer_doc

        _, basic_default, basic_type, basic_doc = get_args_from_cls_or_func(self.get_basic_options)
        self.opt.update(basic_default)
        self.opt.update(self.get_basic_options(train=True))
        self.type_info.update(basic_type)
        self.docs.update(basic_doc)

        # yaml_opt_path = osp.join(_GLOBAL_VARS['stablediffusion'], "configs/autoencoder/autoencoder_kl_64x64x3.yaml")
        yaml_opt_path = osp.join(_GLOBAL_VARS['stablediffusion'], "configs/latent-diffusion/cin-ldm-vq-f8.yaml")
        yaml_opt, yaml_type, yaml_docs = self.get_options_from_yaml(yaml_opt_path)
        self.opt.update(yaml_opt)
        self.type_info.update(yaml_type)
        self.docs.update(yaml_docs)

    def get_basic_options(self, name:str="", resume:str="", train:bool=False, no_test:bool=False, project:str="",
                          debug:bool=False, seed:int=23, postfix:str="", logdir:str="logs", scale_lr:bool=True):
        """
        获取基本参数，如环境参数等

        Args:
            name:       postfix for logdir
            resume:     resume from logdir or checkpoint in logdir
            train:      train
            no_test:    disable test
            project:    name of new or path to existing project
            debug:      enable post-mortem debugging
            seed:       seed for seed_everything
            postfix:    post-postfix for default name
            logdir:     directory for logging dat shit
            scale_lr:   scale base-lr by ngpu * batch_size * n_accumulate

        Returns:

        """
        basic_opt = {}
        basic_opt['name'] = name
        basic_opt['resume'] = resume
        basic_opt['train'] = train
        basic_opt['no_test'] = no_test
        basic_opt['project'] = project
        basic_opt['debug'] = debug
        basic_opt['seed'] = seed
        basic_opt['postfix'] = postfix
        basic_opt['logdir'] = logdir
        basic_opt['scale_lr'] = scale_lr

        return basic_opt

    def get_options_from_yaml(self, yaml_path:str):
        """
        从额外的 yaml 表中更新参数
        """
        assert isinstance(yaml_path, str) and (yaml_path.endswith('.yaml') or yaml_path.endswith('.yml')),\
            f"输入的 yaml_path: {yaml_path} 不是一个有效的yaml文件路径, 请检查！"

        yaml = YAML()
        yaml.preserve_quotes = True
        with open(yaml_path, "r", encoding='utf-8') as f:
            opt_from_yaml = yaml.load(f)

        args_type, args_comment = get_yaml_args_types_comments(yaml_path)

        return opt_from_yaml, args_type, args_comment

    def export_to_yaml(self, yaml_path:str):
        assert isinstance(yaml_path, str) and (yaml_path.endswith('.yaml') or yaml_path.endswith('.yml')),\
            f"输入的 yaml_path: {yaml_path} 不是一个有效的yaml文件路径, 请检查！"

        commented_opt = convert_to_commented_map(self.opt, self.type_info, self.docs)

        yaml = YAML()
        yaml.preserve_quotes = True
        with open(yaml_path, "w", encoding='utf-8') as f:
            yaml.dump(commented_opt, f)



if __name__ == '__main__':
    yaml_path = 'stable_diffusion_second_stage.yml'
    # sd_opt = SDOptionsGenerator()
    # sd_opt.export_to_yaml(yaml_path)

    sd_opt = SDOptions(yaml_path)
    opt = sd_opt.opt

    print(OmegaConf.to_yaml(opt))