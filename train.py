import mindspore as ms
from mindspore import context
import os
import argparse
import yaml
from trainer import ModelTrainer

import numpy as np
import random


# # ===== 固定所有随机种子 =====
"""
随机种子的设置需覆盖以下三个方面，即训练模型时涉及的随机数包括：
框架本身随机数（如参数初始化，这块属于全局随机种子）、
后端环境特别是GPU环境随机数（如cuDNN的随即种子）、
数据加载随机数（如各种数据增强操作，这块属于数据预处理随种子，包括利用python中random模块以及numpy库操作的随机种子）
"""
def set_seed(seed=42):
    # 设置Python环境随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 设置MindSpore全局种子
    ms.set_seed(seed)

    # 启用确定性计算模式（重要）
    context.set_context(deterministic='ON')

    # 设置CUDA确定性选项（GPU环境需要）
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def arg_parser():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="config/FakeDiffer.yml",
                        help="Specified the path of configuration file to be used.")
    return parser.parse_args()


if __name__ == '__main__':
    # 调用固定种子函数
    set_seed(42)

    arg = arg_parser()
    config = arg.config

    with open(config) as config_file:
        # print('37 config_file :', config_file)
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    trainer = ModelTrainer(config, stage="Train")
    trainer.train()
