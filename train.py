import os
from os import path as osp

from omegaconf import OmegaConf

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from configs.options import SDOptions
from ldm.util import instantiate_from_config


def main():
    yaml_path = 'configs/stable_diffusion_first_stage.yml'
    opt_manager = SDOptions(yaml_path=yaml_path)
    opt = opt_manager.opt
    lightning_config = opt_manager.lightning_config
    trainer_config = lightning_config.trainer
    modelckpt_config = opt_manager.default_modelckpt_cfg
    callbacks_config = opt_manager.default_callbacks_cfg
    ngpu = opt_manager.ngpu

    seed_everything(opt.seed)

    try:
        model = instantiate_from_config(opt.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        # TODO 分层管理 logger 配置
        trainer_kwargs["logger"] = instantiate_from_config(opt_manager.logger_config)

        # TODO callbacks yaml表集中生成
        # model checkpoint callback
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            modelckpt_config["params"]["monitor"] = model.monitor
            modelckpt_config["params"]["save_top_k"] = 3
        modelckpt_config = OmegaConf.merge(modelckpt_config, lightning_config.get("modelcheckpoint", OmegaConf.create()))
        print(f"Merged modelckpt-cfg: \n{modelckpt_config}")
        callbacks_config.update({'checkpoint_callback': modelckpt_config})

        # other callbacks
        lightning_callbacks_cfg = lightning_config.callbacks if "callbacks" in lightning_config else OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in lightning_callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": osp.join(opt.ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,
                         'save_weights_only': True
                         }
                     }
            }
            callbacks_config.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_config = OmegaConf.merge(callbacks_config, lightning_callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_config and hasattr(trainer_config, 'resume_from_checkpoint'):
            callbacks_config.ignore_keys_callback.params['ckpt_path'] = trainer_config.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_config:
            del callbacks_config['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_config[k]) for k in callbacks_config]

        trainer_config_dict = OmegaConf.to_container(trainer_config, resolve=True)
        trainer_config_dict.update(trainer_kwargs)
        trainer = Trainer(**trainer_config_dict)
        trainer.logdir = opt.logdir  ### 不太明白为什么要这样写

        # data
        data = instantiate_from_config(opt.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = opt.data.params.batch_size, opt.model.base_learning_rate
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = osp.join(opt.ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = osp.split(opt.logdir)
            dst = osp.join(dst, "debug_runs", name)
            os.makedirs(osp.split(dst)[0], exist_ok=True)
            os.rename(opt.logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())

    #     print(1)



if __name__ == '__main__':
    main()