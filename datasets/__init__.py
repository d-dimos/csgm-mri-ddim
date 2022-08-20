from dataloaders import *


def get_dataloader(files, config, args):
    dataset = MVU_Estimator_Brain(files,
                                  input_dir=args.data_dir,
                                  maps_dir=args.maps_dir,
                                  project_dir='',
                                  image_size=config.image_size,
                                  R=config.R,
                                  pattern=config.pattern,
                                  orientation=config.orientation)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=config.sampling.batch_size,
                                         sampler=None,
                                         shuffle=False)

    return loader
