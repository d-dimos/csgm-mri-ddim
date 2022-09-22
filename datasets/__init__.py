from .dataloaders import *


def get_dataloader(files, config, args):
    if args.anatomy == 'brain':
        dataset = MVU_Estimator_Brain(files,
                                      input_dir=config.data_dir,
                                      maps_dir=config.maps_dir,
                                      project_dir='',
                                      image_size=config.image_size,
                                      R=args.R,
                                      pattern=args.pattern,
                                      orientation=args.orientation)

    else:  # args.anatomy == knee
        dataset = MVU_Estimator_Knees(files,
                                      input_dir=config.data_dir,
                                      maps_dir=config.maps_dir,
                                      project_dir='',
                                      image_size=config.image_size,
                                      R=args.R,
                                      pattern=args.pattern,
                                      orientation=args.orientation)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=config.batch_size,
                                         sampler=None,
                                         shuffle=False)

    return loader
