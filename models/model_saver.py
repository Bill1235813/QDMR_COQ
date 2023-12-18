import torch


class ModelSaver(object):
    """Simple model saver to filesystem"""

    def __init__(self, base_path, model, optim, logger):
        self.base_path = base_path
        self.model = model
        self.optim = optim
        self.logger = logger

    def save(self, tp=""):
        model_state_dict = self.model.state_dict()

        checkpoint = {
            'model': model_state_dict,
            'optim': self.optim.state_dict(),
        }

        self.logger.info("Saving checkpoint %s_%s.pt" % (self.base_path, tp))
        checkpoint_path = '%s_%s.pt' % (self.base_path, tp)
        torch.save(checkpoint, checkpoint_path)
