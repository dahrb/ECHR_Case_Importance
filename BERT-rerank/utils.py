from sklearn.metrics import matthews_corrcoef
import torch
from transformers import AutoTokenizer
from sentence_transformers.cross_encoder import CrossEncoder
import logging
import os
from functools import wraps
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, is_torch_npu_available
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import PushToHubMixin

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import fullname, get_device_name, import_from_string

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=4, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model = model
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model
            self.counter = 0

class CrossEncoderMod(CrossEncoder):

    """
    Modified implementation of CrossEncoder from Sentence-Transformers library

    Adds Early Stopping functionality based on Average Precision
    """

    def __init__(self,
        model_name: str,
        num_labels: int = None,
        max_length: int = None,
        device: str = None,
        tokenizer_args: Dict = None,
        automodel_args: Dict = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        default_activation_function=None,
        classifier_dropout: float = None) -> None:
        
        super().__init__(model_name,num_labels,max_length,device,tokenizer_args,automodel_args,trust_remote_code,revision,local_files_only,default_activation_function,classifier_dropout)

    def fit(
        self,
        train_dataloader: DataLoader,
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        loss_fct=None,
        activation_fct=nn.Identity(),
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
    ) -> None:
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        Args:
            train_dataloader (DataLoader): DataLoader with training InputExamples
            evaluator (SentenceEvaluator, optional): An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc. Defaults to None.
            epochs (int, optional): Number of epochs for training. Defaults to 1.
            loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss(). Defaults to None.
            activation_fct: Activation function applied on top of logits output of model.
            scheduler (str, optional): Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts. Defaults to "WarmupLinear".
            warmup_steps (int, optional): Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero. Defaults to 10000.
            optimizer_class (Type[Optimizer], optional): Optimizer. Defaults to torch.optim.AdamW.
            optimizer_params (Dict[str, object], optional): Optimizer parameters. Defaults to {"lr": 2e-5}.
            weight_decay (float, optional): Weight decay for model parameters. Defaults to 0.01.
            evaluation_steps (int, optional): If > 0, evaluate the model using evaluator after each number of training steps. Defaults to 0.
            output_path (str, optional): Storage path for the model and evaluation files. Defaults to None.
            save_best_model (bool, optional): If true, the best model (according to evaluator) is stored at output_path. Defaults to True.
            max_grad_norm (float, optional): Used for gradient normalization. Defaults to 1.
            use_amp (bool, optional): Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0. Defaults to False.
            callback (Callable[[float, int, int], None], optional): Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`. Defaults to None.
            show_progress_bar (bool, optional): If True, output a tqdm progress bar. Defaults to True.
        """
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            if is_torch_npu_available():
                scaler = torch.npu.amp.GradScaler()
            else:
                scaler = torch.cuda.amp.GradScaler()
        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(
                optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
            )

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        skip_scheduler = False

        early_stopping = EarlyStopping(patience=3, delta=0.01)

        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(
                train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar
            ):
                if use_amp:
                    with torch.autocast(device_type=self._target_device.type):
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(
                        evaluator, output_path, save_best_model, epoch, training_steps, callback
                    )

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                score = self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)
                logger.info(f"Score {score}")

                # Check early stopping
                early_stopping(score, self.model)
                if early_stopping.early_stop:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    self.epoch = epoch + 1
                    break

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback) -> None:
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

            return score            


def calculate_mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)

# Custom collate function
def custom_collate_fn(batch):
    texts = [example.texts for example in batch]
    labels = [example.label for example in batch]
    
    model_name = "nlpaueb/legal-bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize texts
    tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.float)
    
    return tokenized_texts, labels
