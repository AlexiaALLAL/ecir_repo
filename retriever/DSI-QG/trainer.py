import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
from torch.utils.data import Dataset
import torch


class DSITrainer(Trainer):
    def __init__(self, restrict_decode_vocab, id_max_length, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = id_max_length

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs['labels'] = inputs['labels'].to(self.args.device)
        with torch.no_grad():
            # Greedy search
            # doc_ids = model.generate(
            #     inputs['input_ids'].to(self.args.device),
            #     max_length=20,
            #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            #     early_stopping=True,)

            # Beam search
            batch_beams = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=20,
                num_beams=20,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                num_return_sequences=20,
                early_stopping=True, )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)

            inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 20, -1)

        return (None, batch_beams, inputs['labels'])

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.processing_class is not None and hasattr(self.processing_class, "pad_token_id"):
            pad_token_id = self.processing_class.pad_token_id
        else:
            pad_token_id = -100
        tensor[tensor == -100] = self.processing_class.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


class DocTqueryTrainer(Trainer):
    def __init__(self, do_generation: bool, **kwds):
        super().__init__(**kwds)
        self.do_generation = do_generation
        self.stored_docids = []  # Store docids separately for generation task

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.do_generation:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        outputs = self.model.generate(
            input_ids=inputs[0]['input_ids'].to(self.args.device),
            attention_mask=inputs[0]['attention_mask'].to(self.args.device),
            max_length=self.max_length,
            do_sample=True,
            top_k=self.top_k,
            num_return_sequences=self.num_return_sequences)
        
        # Store docids separately (can be strings or integers) and repeat for multiple sequences
        docids = [docid for docid in inputs[1] for _ in range(self.num_return_sequences)]
        self.stored_docids.extend(docids)

        if outputs.shape[-1] < self.max_length:
            outputs = self._pad_tensors_to_max_len(outputs, self.max_length)
        return (None, outputs.reshape(inputs[0]['input_ids'].shape[0], self.num_return_sequences, -1),
                None)  # Return None for labels to avoid padding issues

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.processing_class is not None and hasattr(self.processing_class, "pad_token_id"):
            pad_token_id = self.processing_class.pad_token_id
        else:
            pad_token_id = -100
        tensor[tensor == -100] = self.processing_class.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def predict(
            self,
            test_dataset: Dataset,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            max_length: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            top_k: Optional[int] = None,
    ):

        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.top_k = top_k
        self.stored_docids = []  # Reset stored docids before prediction
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
