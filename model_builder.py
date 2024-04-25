"""
This file is for models creation
"""
import torch
from transformers import AutoTokenizer, AutoConfig
from models.single_model import SingleRunModel
from utils.optimizers import Optimizer


def load_train_model(opt, logger):
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
    else:
        checkpoint = None
    model = build_model(opt, checkpoint, logger)
    model.train()
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)
    return model, optim

def load_test_model(opt, logger):
    logger.info('Loading checkpoint from %s' % opt.model_path)
    checkpoint = torch.load(opt.model_path,
                            map_location=lambda storage, loc: storage)
    model = build_model(opt, checkpoint, logger)
    model.eval()
    return model


def build_model(model_opt, checkpoint, logger):
    logger.info('Building model...')
    if model_opt.architect == "longt5":
        model_name = 'google/long-t5-tglobal-base'
    else:
        model_name = model_opt.architect

    # Build SingleRunModel
    device = torch.device("cuda")
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = SingleRunModel(config=config)
    model.tokenizer = tokenizer

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)

    model.to(device)
    return model
