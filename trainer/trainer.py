""" In charge of training details, loss compute, and statistics. """
import numpy as np
import torch
import math
from utils.statistics import Statistics
from utils.earlystopping import EarlyStopping
from utils.report_manager import ReportMgr
from models.model_saver import ModelSaver


class Trainer(object):
    """ Class that controls the training process. """

    def __init__(self, opt, model, optim, logger):
        # Basic attributes.
        self.model = model
        self.optim = optim
        self.logger = logger
        self.accum_count = opt.accum_count
        self.batch_size = opt.batch_size
        self.device = torch.device("cuda:%s" % 0)
        self.trim_size = opt.trim_size
        self.loss_ignore_index = -100

        self.model_saver = ModelSaver(opt.save_model, model, optim, logger)
        self.earlystopper = EarlyStopping(
            opt.early_stopping, logger, opt.save_model
        ) if opt.early_stopping > 0 else None
        self.report_manager = ReportMgr(opt, logger)

        # Set model in training mode.
        self.model.train()
        self.last_train_batch = 0
        self.last_test_batch = 0
        self.sample_ids = None

    def totensor_batch(self, batch):
        source = [d[0] for d in batch]
        target = [d[1] for d in batch]
        inputs = self.model.tokenizer(source, return_tensors="pt",
                                      max_length=self.trim_size,
                                      truncation=True, padding=True)
        input_ids = inputs.input_ids[:, :self.trim_size].to(
            self.device, non_blocking=True)
        attn_mask = inputs.attention_mask[:, :self.trim_size].to(
            self.device, non_blocking=True)
        labels = self.model.tokenizer(target, return_tensors="pt",
                                      max_length=self.trim_size,
                                      truncation=True, padding=True)
        dinput_ids = labels.input_ids[:, :self.trim_size].to(
            self.device, non_blocking=True)
        dattn_mask = labels.attention_mask[:, :self.trim_size].to(
            self.device, non_blocking=True)

        return input_ids, attn_mask, dinput_ids, dattn_mask

    def process_batch(self, data, train=True):
        batch_ids = None
        if train:
            last_batch = self.last_train_batch
        else:
            last_batch = self.last_test_batch

        batch_size = self.batch_size
        if (last_batch + batch_size) <= len(data):
            next_batch = last_batch + batch_size
            if train:
                batch_ids = self.sample_ids[last_batch:next_batch]
                batch = [data[d] for d in batch_ids]
            else:
                batch = data[last_batch:next_batch]
        else:
            if train:
                batch_ids = self.sample_ids[last_batch:]
                self.sample_ids = np.random.permutation(len(data))
                batch_ids = np.concatenate(
                    (batch_ids,
                     self.sample_ids[:(last_batch + batch_size) - len(data)])
                )
                batch = [data[d] for d in batch_ids]
                next_batch = (last_batch + batch_size) - len(data)
            else:
                batch = data[last_batch:]
                next_batch = 0

        if train:
            self.last_train_batch = next_batch
        else:
            self.last_test_batch = next_batch

        return self.totensor_batch(batch), batch_ids

    def train(self,
              train_data,
              train_steps=30000,
              valid_data=None,
              valid_steps=10000):
        if valid_data is None:
            self.logger.info('Start training loop without validation...')
        else:
            self.logger.info(
                'Start training loop and validate every %d steps...',
                valid_steps)

        report_stats = Statistics(self.logger)
        self._start_report_manager(report_stats.start_time)
        self.sample_ids = np.random.permutation(len(train_data))
        best_acc = 0
        step = self.optim.training_step

        for i in range(step, train_steps):
            step = self.optim.training_step
            self._gradient_accumulation(
                train_data, self.batch_size * self.accum_count, report_stats
            )
            report_stats = self._maybe_report_training(step, train_steps,
                                                       self.optim.learning_rate(),
                                                       report_stats)

            if valid_data is not None and step % valid_steps == 0:
                valid_stats = self.validate(valid_data)
                self._report_step(self.optim.learning_rate(), step,
                                  valid_stats=valid_stats)
                if valid_stats.accuracy() > best_acc:
                    best_acc = valid_stats.accuracy()
                    print("Best Acc: %s" % best_acc)
                    self.model_saver.save("best")
                else:
                    self.model_saver.save("last")
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    if self.earlystopper.has_stopped():
                        break

    def validate(self, valid_data):
        self.model.eval()
        self.last_test_batch = 0

        with torch.no_grad():
            stats = Statistics(self.logger)
            iters = math.ceil(len(valid_data) / self.batch_size)

            for i in range(iters):
                (input_ids, attn_mask, dinput_ids, dattn_mask), _ = \
                    self.process_batch(valid_data, train=False)
                label = dinput_ids[1:]
                label[label == 0] = self.loss_ignore_index
                output_dict = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    decoder_input_ids=dinput_ids,
                    decoder_attention_mask=dattn_mask,
                    labels=label,
                    return_dict=True
                )
                stats.update(self._compute_loss_stats(
                    output_dict[0], output_dict[1], label
                ))

        self.model.train()
        return stats

    def _gradient_accumulation(self, train_data, report_stats):
        self.optim.zero_grad()

        for k in range(self.accum_count):
            (input_ids, attn_mask, dinput_ids, dattn_mask), _ \
                = self.process_batch(train_data, train=True)
            label = dinput_ids[1:]
            label[label == 0] = self.loss_ignore_index
            output_dict = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                decoder_input_ids=dinput_ids,
                decoder_attention_mask=dattn_mask,
                labels=label,
                return_dict=True
            )
            report_stats.update(self._compute_loss_stats(
                output_dict[0], output_dict[1], label
            ))
            output_dict[0].backward()

        self.optim.step()

    def _start_report_manager(self, start_time):
        if self.report_manager is not None:
            self.report_manager.start_time = start_time

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _compute_loss_stats(self, loss, scores, target):
        pred = scores.max(2)[1]
        non_padding = target.ne(self.loss_ignore_index)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return Statistics(loss.item(), num_non_padding, num_correct)
