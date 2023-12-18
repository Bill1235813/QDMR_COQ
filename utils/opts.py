""" Implementation of all available options """
from __future__ import print_function


def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False, type=bool,
               default=True, help='config file save path')


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group('Model')
    group.add('--trim_size', '-trim_size', type=int, default=512,
              help='Trimed sent length for transformer inputs.')
    group.add('--architect', '-architect', type=str, default='t5-base')
    group.add('--model_dtype', '-model_dtype', default='fp32',
              choices=['fp32', 'fp16'],
              help='Data type of the model.')


def train_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('General')

    group.add('--task_name', '-task_name', type=str, default=None)
    group.add('--train_data', '-train_data', required=True)
    group.add('--train_datat', '-train_datat', required=True)
    group.add('--valid_data', '-valid_data', type=str, default=None)
    group.add('--valid_datat', '-valid_datat', type=str, default=None)

    group.add('--save_model', '-save_model', default='model',
              help="Model filename (the model will be saved as "
                   "<save_model>_N.pt where N is the number "
                   "of steps")
    group.add('--save_checkpoint_steps', '-save_checkpoint_steps',
              type=int, default=50000,
              help="""Save a checkpoint every X steps""")

    # # GPU
    # group.add('--gpu_ranks', '-gpu_ranks', default=[], nargs='*', type=int,
    #           help="list of ranks of each process.")
    # group.add('--world_size', '-world_size', default=1, type=int,
    #           help="total number of distributed processes.")
    # group.add('--gpu_backend', '-gpu_backend',
    #           default="nccl", type=str,
    #           help="Type of torch distributed backend")
    # group.add('--master_ip', '-master_ip', default="localhost", type=str,
    #           help="IP of master for torch.distributed training.")
    # group.add('--master_port', '-master_port', default=10000, type=int,
    #           help="Port of master for torch.distributed training.")
    # group.add('--queue_size', '-queue_size', default=40, type=int,
    #           help="Size of queue for each process in producer/consumer")

    group.add('--seed', '-seed', type=int, default=-1,
              help="Random seed used for the experiments "
                   "reproducibility.")

    # Init options
    group = parser.add_argument_group('Initialization')
    group.add('--train_from', '-train_from', default='', type=str,
              help="If training from a checkpoint then this is the "
                   "path to the pretrained model's state_dict.")
    group.add('--reset_optim', '-reset_optim', default='none',
              choices=['none', 'all', 'states', 'keep_states'],
              help="Optimization resetter when train_from.")

    # Optimization options
    group = parser.add_argument_group('Optimization-Type')
    group.add('--batch_size', '-batch_size', type=int, default=64,
              help='Maximum batch size for training')
    group.add('--accum_count', '-accum_count', type=int, default=1,
              help="Accumulate gradient this many times. "
                   "Approximately equivalent to updating "
                   "batch_size * accum_count batches at once. ")
    group.add('--valid_steps', '-valid_steps', type=int, default=10000,
              help='Perfom validation every X steps')
    group.add('--valid_batch_size', '-valid_batch_size', type=int, default=32,
              help='Maximum batch size for validation')
    group.add('--max_generator_batches', '-max_generator_batches',
              type=int, default=32,
              help="Maximum batches of words in a sequence to run "
                   "the generator on in parallel. Higher is faster, but "
                   "uses more memory. Set to 0 to disable.")
    group.add('--train_steps', '-train_steps', type=int, default=100000,
              help='Number of training steps')
    group.add('--stop_steps', '-stop_steps', type=int, default=3000,
              help='Number of stoping steps')
    group.add('--early_stopping', '-early_stopping', type=int, default=0,
              help='Number of validation steps without improving.')
    group.add('--optim', '-optim', default='sgd',
              choices=['sgd', 'adagrad', 'adadelta', 'adam',
                       'sparseadam', 'adafactor', 'fusedadam'],
              help="Optimization method.")
    group.add('--adagrad_accumulator_init', '-adagrad_accumulator_init',
              type=float, default=0,
              help="Initializes the accumulator values in adagrad. "
                   "Mirrors the initial_accumulator_value option "
                   "in the tensorflow adagrad (use 0.1 for their default).")
    group.add('--max_grad_norm', '-max_grad_norm', type=float, default=5,
              help="If the norm of the gradient vector exceeds this, "
                   "renormalize it to have the norm equal to "
                   "max_grad_norm")
    group.add('--adam_beta1', '-adam_beta1', type=float, default=0.9,
              help="The beta1 parameter used by Adam. ")
    group.add('--adam_beta2', '-adam_beta2', type=float, default=0.999,
              help='The beta2 parameter used by Adam. ')
    group.add('--label_smoothing', '-label_smoothing', type=float, default=0.0,
              help="Label smoothing value epsilon. "
                   "Probabilities of all non-true labels "
                   "will be smoothed by epsilon / (vocab_size - 1). "
                   "Set to zero to turn off label smoothing. "
                   "For more detailed information, see: "
                   "https://arxiv.org/abs/1512.00567")

    # Learning rate
    group = parser.add_argument_group('Optimization-Rate')
    group.add('--learning_rate', '-learning_rate', type=float, default=1.0,
              help="Starting learning rate. "
                   "Recommended settings: sgd = 1, adagrad = 0.1, "
                   "adadelta = 1, adam = 0.001")
    group.add('--learning_rate_decay', '-learning_rate_decay',
              type=float, default=0.5,
              help="If update_learning_rate, decay learning rate by "
                   "this much if steps have gone past "
                   "start_decay_steps")
    group.add('--start_decay_steps', '-start_decay_steps',
              type=int, default=50000,
              help="Start decaying every decay_steps after "
                   "start_decay_steps")
    group.add('--decay_steps', '-decay_steps', type=int, default=10000,
              help="Decay every decay_steps")
    group.add('--decay_method', '-decay_method', type=str, default="none",
              choices=['noam', 'noamwd', 'rsqrt', 'none'],
              help="Use a custom decay rate.")
    group.add('--warmup_steps', '-warmup_steps', type=int, default=4000,
              help="Number of warmup steps for custom decay.")

    group = parser.add_argument_group('Logging')
    group.add('--report_every', '-report_every', type=int, default=50,
              help="Print stats at this interval.")
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    # Use Tensorboard for visualization during training
    group.add('--tensorboard', '-tensorboard', action="store_true",
              help="Use tensorboard for visualization during training. "
                   "Must have the library tensorboard >= 1.14.")


def translate_opts(parser):
    """ Translation / inference options """
    group = parser.add_argument_group('Model')
    group.add('--model_path', '-model_path', type=str, required=True,
              help="Path to the model .pt file.")
    group.add('--fp32', '-fp32', action='store_true',
              help="Force the model to be in FP32 "
                   "because FP16 is very slow on GTX1080(ti).")

    group = parser.add_argument_group('Data')
    group.add('--data_type', '-data_type', default="text",
              help="Type of the source input. Options: [text|img].")

    group.add('--src', '-src', required=True,
              help="Source sequence to decode (one line per "
                   "sequence)")
    group.add('--src_dir', '-src_dir', default="",
              help='Source directory for image or audio files')
    group.add('--tgt', '-tgt',
              help='True target sequence (optional)')
    group.add('--target_vocab', '-target_vocab', type=str, default=None)
    group.add('--output', '-output', default='pred.txt',
              help="Path to output the predictions (each line will "
                   "be the decoded sequence")
    group.add('--report_time', '-report_time', action='store_true',
              help="Report some translation time metrics")

    group = parser.add_argument_group('Random-Sampling')
    group.add('--random_sampling_topk', '-random_sampling_topk',
              default=1, type=int,
              help="Set this to -1 to do random sampling from full "
                   "distribution. Set this to value k>1 to do random "
                   "sampling restricted to the k most likely next tokens. "
                   "Set this to 1 to use argmax or for doing beam "
                   "search.")
    group.add('--random_sampling_temp', '-random_sampling_temp',
              default=1., type=float,
              help="If doing random sampling, divide the logits by "
                   "this before computing softmax during decoding.")
    group.add('--seed', '-seed', type=int, default=829,
              help="Random seed")

    group = parser.add_argument_group('Beam')
    group.add('--beam_size', '-beam_size', type=int, default=5,
              help='Beam size')
    group.add('--min_length', '-min_length', type=int, default=0,
              help='Minimum prediction length')
    group.add('--max_length', '-max_length', type=int, default=100,
              help='Maximum prediction length.')

    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    group.add('--stepwise_penalty', '-stepwise_penalty', action='store_true',
              help="Apply penalty at every decoding step. "
                   "Helpful for summary penalty.")
    group.add('--length_penalty', '-length_penalty', default='none',
              choices=['none', 'wu', 'avg'],
              help="Length Penalty to use.")
    group.add('--ratio', '-ratio', type=float, default=-0.,
              help="Ratio based beam stop condition")
    group.add('--coverage_penalty', '-coverage_penalty', default='none',
              choices=['none', 'wu', 'summary'],
              help="Coverage Penalty to use.")
    group.add('--alpha', '-alpha', type=float, default=0.,
              help="Google NMT length penalty parameter "
                   "(higher = longer generation)")
    group.add('--beta', '-beta', type=float, default=-0.,
              help="Coverage penalty parameter")
    group.add('--block_ngram_repeat', '-block_ngram_repeat',
              type=int, default=0,
              help='Block repetition of ngrams during decoding.')
    group.add('--ignore_when_blocking', '-ignore_when_blocking',
              nargs='+', type=str, default=[],
              help="Ignore these strings when blocking repeats. "
                   "You want to block sentence delimiters.")
    group.add('--replace_unk', '-replace_unk', action="store_true",
              help="Replace the generated UNK tokens with the "
                   "source token that had highest attention weight. If "
                   "phrase_table is provided, it will look up the "
                   "identified source token and give the corresponding "
                   "target token. If it is not provided (or the identified "
                   "source token does not exist in the table), then it "
                   "will copy the source token.")

    group = parser.add_argument_group('Logging')
    group.add('--verbose', '-verbose', action="store_true",
              help='Print scores and predictions for each sentence')
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--dump_beam', '-dump_beam', type=str, default="",
              help='File to dump beam information to.')
    group.add('--n_best', '-n_best', type=int, default=1,
              help="If verbose is set, will output the n_best "
                   "decoded sentences")

    group = parser.add_argument_group('Efficiency')
    group.add('--batch_size', '-batch_size', type=int, default=30,
              help='Batch size')
    group.add('--batch_type', '-batch_type', default='sents',
              choices=["sents", "tokens"],
              help="Batch grouping for batch_size. Standard "
                   "is sents. Tokens will do dynamic batching")
    # group.add('--gpu', '-gpu', type=int, default=-1,
    #           help="Device to run on")
