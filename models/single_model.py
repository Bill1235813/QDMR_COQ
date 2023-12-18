from transformers import AutoModelForPreTraining


class SingleRunModel(AutoModelForPreTraining):
    """
    A model for single run QA. Could be T5, longT5, etc.
    """

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = None
