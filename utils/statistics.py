""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys


class Statistics(object):
    """
    Accumulator for loss statistics. Currently, it calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, logger, loss=0, n_words=0, n_correct=0):
        self.logger = logger
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.start_time = time.time()

    def update(self, stat):
        """
        Update statistics by summing values with another `Statistics` object

        Args:
            stat: another statistic object
        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """ write out statistics to stdout """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        self.logger.info(
            ("Step %s; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
             "lr: %7.5f; %3.0f tok/s; %6.0f sec")
            % (step_fmt, self.accuracy(), self.ppl(), self.xent(),
               learning_rate, self.n_words / (t + 1e-5), time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
