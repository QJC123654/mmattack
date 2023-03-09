import mmcv
from mmcv.runner import load_checkpoint

# from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
def load_model(config:str='configs/gfl/gfl_r50_fpn_1x_coco.py',
        checkpoint:str='checkpoints/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth', device:str='cuda:0'):
    # Choose to use a config and initialize the detector
    # Setup a checkpoint file to load
    # Set the device to be used for evaluation
    # Load the config
    config = mmcv.Config.fromfile(config)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None

    # Initialize the detector
    model = build_detector(config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = config

    # Convert the model to GPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()
    return model