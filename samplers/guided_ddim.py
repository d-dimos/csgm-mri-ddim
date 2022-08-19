import os

from utils import get_all_files

__all__ = ['guided_DDIM']


# def get_model(config):
#     if config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA':
#         return NCSNv2(config).to(config.device)
#     elif config.data.dataset == "FFHQ":
#         return NCSNv2Deepest(config).to(config.device)
#     elif config.data.dataset == 'LSUN':
#         return NCSNv2Deeper(config).to(config.device)


class guided_DDIM:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.files = get_all_files(args.data_dir, pattern='*.h5')

        os.makedirs(args.log_path, exist_ok=True)

    def sample(self):
        print("DDIM sampling")

