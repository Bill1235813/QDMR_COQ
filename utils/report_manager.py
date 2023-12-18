""" Report manager utility """
import os
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.statistics import Statistics


class ReportMgr(object):
    def __init__(self, opt, logger):
        self.report_every = opt.report_every
        self.start_time = -1.0
        if opt.tensorboard:
            tensorboard_log_dir = opt.save_model
            if not opt.train_from:
                tensorboard_log_dir += datetime.now().strftime(
                    "/%b-%d_%H-%M-%S")
            else:
                for name in os.listdir(tensorboard_log_dir):
                    if os.path.isdir(name):
                        break
                tensorboard_log_dir = "%s/%s" % (tensorboard_log_dir, name)

            self.tensorboard_writer = SummaryWriter(tensorboard_log_dir)
        else:
            self.tensorboard_writer = None
        self.logger = logger

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def maybe_log_tensorboard(self, stats, prefix, learning_rate, step):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(
                prefix, self.tensorboard_writer, learning_rate, step)

    def report_training(self, step, num_steps, learning_rate, report_stats):
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if step % self.report_every == 0:
            report_stats.output(step, num_steps, learning_rate, self.start_time)
            self.maybe_log_tensorboard(report_stats, "progress", learning_rate,
                                       step)
            return Statistics(self.logger)
        else:
            return report_stats

    def report_step(self, lr, step, train_stats=None, valid_stats=None):
        if train_stats is not None:
            self.log('Train xent: %g' % train_stats.xent())
            self.log('Train accuracy: %g' % train_stats.accuracy())
            self.maybe_log_tensorboard(train_stats, "train", lr, step)

        if valid_stats is not None:
            self.log('Validation xent: %g' % valid_stats.xent())
            self.log('Validation accuracy: %g' % valid_stats.accuracy())
            self.maybe_log_tensorboard(valid_stats, "valid", lr, step)
