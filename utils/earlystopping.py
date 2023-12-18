import os

from enum import Enum
from shutil import copyfile


class PatienceEnum(Enum):
    IMPROVING = 0
    DECREASING = 1
    STOPPED = 2


class AccuracyScorer(object):
    def __init__(self):
        self.best_score = float("-inf")
        self.name = "acc"

    def is_improving(self, stats):
        raise stats.accuracy() > self.best_score

    def is_decreasing(self, stats):
        raise stats.accuracy() < self.best_score

    def update(self, stats):
        self.best_score = stats.accuracy()


class EarlyStopping(object):

    def __init__(self, tolerance, logger, base_path):
        """
        Callable class to keep track of early stopping.

        Args:
            tolerance(int): number of validation steps without improving
        """

        self.tolerance = tolerance
        self.stalled_tolerance = self.tolerance
        self.current_tolerance = self.tolerance
        self.scorer = AccuracyScorer()
        self.status = PatienceEnum.IMPROVING
        self.current_step_best = 0
        self.save_steps = 500
        self.logger = logger
        self.base_path = base_path

    def __call__(self, valid_stats, step):
        """
        Update the internal state of early stopping mechanism, whether to
        continue training or stop the train procedure.

        Checks whether the scores from all pre-chosen scorers improved. If
        every metric improve, then the status is switched to improving and the
        tolerance is reset. If every metric deteriorate, then the status is
        switched to decreasing and the tolerance is also decreased; if the
        tolerance reaches 0, then the status is changed to stopped.
        Finally, if some improved and others not, then it's considered stalled;
        after tolerance number of stalled, the status is switched to stopped.

        :param valid_stats: Statistics of dev set
        """

        if self.status == PatienceEnum.STOPPED:
            # Don't do anything
            return

        if self.scorer.is_improving(valid_stats):
            self._update_increasing(valid_stats, step)

        elif self.scorer.is_decreasing(valid_stats):
            self._update_decreasing()
        else:
            self._update_stalled()

    def _update_stalled(self):
        self.stalled_tolerance -= 1

        self.logger.info(
            "Stalled patience: {}/{}".format(self.stalled_tolerance,
                                             self.tolerance))

        if self.stalled_tolerance == 0:
            self.logger.info(
                "Training finished after stalled validations. Early Stop!"
            )

        self._decreasing_or_stopped_status_update(self.stalled_tolerance)

    def _update_increasing(self, valid_stats, step):
        self.current_step_best = step
        self.logger.info(
            "Model is improving {}: {:g} --> {:g}.".format(
                self.scorer.name, self.scorer.best_score,
                valid_stats.accuracy())
        )
        self.scorer.update(valid_stats)

        # Reset tolerance
        self.current_tolerance = self.tolerance
        self.stalled_tolerance = self.tolerance

        # Update current status
        self.status = PatienceEnum.IMPROVING

    def _update_decreasing(self):
        # Decrease tolerance
        self.current_tolerance -= 1

        # Log
        self.logger.info(
            "Decreasing patience: {}/{}".format(self.current_tolerance,
                                                self.tolerance)
        )
        # Log
        if self.current_tolerance == 0:
            self.logger.info(
                "Training finished after not improving. Early Stop!")

        self._decreasing_or_stopped_status_update(self.current_tolerance)

    def log_best_step(self):
        self.logger.info("Best model found at step {}".format(
            self.current_step_best))

    def _remove_files(self):
        self.logger.info("Removing Files")
        full_steps = self.current_step_best + (
                    self.tolerance + 1) * self.save_steps
        for i in range(0, full_steps, self.save_steps):
            old_path = "{}_step_{}.pt".format(self.base_path, i)
            if os.path.isfile(old_path):
                os.remove(old_path)

    def _decreasing_or_stopped_status_update(self, tolerance):
        self.status = PatienceEnum.DECREASING if tolerance > 0 \
            else PatienceEnum.STOPPED

    def has_stopped(self):
        return self.status == PatienceEnum.STOPPED
