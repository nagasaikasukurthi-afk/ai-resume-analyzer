import torch

from transformers import Trainer


class WeightedTrainer(Trainer):

    def __init__(
        self,
        *args,
        class_weights=None,
        **kwargs
    ):

        super().__init__(
            *args,
            **kwargs
        )

        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.get("labels")

        outputs = model(
            **inputs
        )

        logits = outputs.get(
            "logits"
        )

        loss_function = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(
                logits.device
            ),
            ignore_index=-100
        )

        loss = loss_function(
            logits.view(-1, model.config.num_labels),
            labels.view(-1)
        )

        return (
            loss,
            outputs
        ) if return_outputs else loss