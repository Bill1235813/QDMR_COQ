"""Training on a single process."""
# import sys
# sys.path.append("./")
from utils.opts import config_opts, model_opts, train_opts
from utils.misc import set_random_seed, print_model_parameters
from utils.logging import init_logger
from utils.parse import ArgumentParser
from trainer.trainer import Trainer
from model_builder import load_train_model


def load_dataset(task, source_file, target_file):
    source_bos = task + ": "
    target_bos = "<pad> "
    with open(source_file) as f:
        source_data = f.read().splitlines()
    with open(target_file) as f:
        target_data = f.read().splitlines()
    return [(source_bos + d[0], target_bos + d[1]) for d in
            zip(source_data, target_data)]


def single_train(opt):
    # Init seeds and logger
    set_random_seed(opt.seed, False)
    logger = init_logger(opt.log_file)
    logger.info('Args: {}'.format(opt))

    # Build model
    model, optim = load_train_model(opt, logger)
    print_model_parameters(model, logger)

    train_data = load_dataset(opt.task_name, opt.architect, opt.train_data,
                              opt.train_datat)
    valid_data = load_dataset(opt.task_name, opt.architect, opt.valid_data,
                              opt.valid_datat)

    trainer = Trainer(opt, model, optim, logger)

    # preprocess maml
    logger.info('Start normal training.')

    trainer.train(
        train_data,
        train_steps=opt.train_steps,
        valid_data=valid_data,
        valid_steps=opt.valid_steps
    )


def get_opt():
    parser = ArgumentParser(description='train.py')

    config_opts(parser)
    model_opts(parser)
    train_opts(parser)
    opt = parser.parse_with_config()

    parser.save_config(opt)
    return opt


def main():
    opt = get_opt()
    single_train(opt)


if __name__ == "__main__":
    main()
