import time

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torchmetrics import Metric


class PLModel(pl.LightningModule):
    '''
    The on_ functions are hooks used to link into a section
    of the pl trainer.fit() function
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._train_time = 0.0
        self._val_time = 0.0
        self._test_time = 0.0
        self._train_start_time = time.time()
        self._val_start_time = time.time()
        self._test_start_time = time.time()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self._train_time = checkpoint["train_time"]
        self._val_time = checkpoint["val_time"]

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["train_time"] = self._train_time
        checkpoint["val_time"] = self._val_time

    def on_train_epoch_start(self) -> None:
        self._train_start_time = time.time()

    def on_train_epoch_end(self, unused=None) -> None:
        elapsed = time.time() - self._train_start_time
        self._train_time += elapsed
        self.log_dict(
            {
                "time/train_epoch_time": elapsed,
                "time/train_total_time": self._train_time,
            },
            on_epoch=True,
            logger=True,
        )

    def on_validation_epoch_start(self) -> None:
        self._val_start_time = time.time()

    def on_validation_epoch_end(self) -> None:
        elapsed = time.time() - self._val_start_time
        self._val_time += elapsed
        res = self._accumulate_data(self.validation_step_outputs)

        self.log_dict(
            {f"val/{key}": val for key, val in res.items() if key != "total"},
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )

        self.log_dict(
            {
                "time/val_epoch_time": elapsed,
                "time/val_total_time": self._val_time,
            },
            on_epoch=True,
            logger=True,
        )
        self.validation_step_outputs.clear()  # Freeing memory

    def on_test_epoch_start(self) -> None:
        self._test_start_time = time.time()

    def on_test_epoch_end(self, outputs) -> None:
        # self.on_validation_epoch_end(outputs)
        elapsed = time.time() - self._test_start_time
        self._test_time += elapsed
        res = self._accumulate_data(self.test_step_outputs)

        self.log_dict(
            {f"val/{key}": val for key, val in res.items() if key != "total"},
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )

        self.log_dict(
            {
                "time/test_epoch_time": elapsed,
                "time/test_total_time": self._test_time,
            },
            on_epoch=True,
            logger=True,
        )
        self.test_step_outputs.clear()

    def _accumulate_data(self, outputs):
        keys = [key for key in outputs[0].keys() if key != "size"]

        total = torch.zeros(1, dtype=torch.int, device=self.device)
        res = {
            key: torch.zeros(1, dtype=outputs[0][key].dtype, device=self.device)
            for key in keys
        }
        for batch in outputs:
            size = batch["size"]
            total += size
            for key in keys:
                res[key] += batch[key] * size

        total = torch.sum(total)
        for key in keys:
            res[key] = torch.sum(res[key])
            res[key] /= total

        return {**res, **{"total": total}}

    def training_step(self, batch, batch_index):
        res = self._evaluate(batch, batch_index)
        self.log_dict(
            {f"train/{key}": val for key, val in res.items()},
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            reduce_fx="mean",
        )
        self.log_dict(
            {
                "time/elapsed": time.time()
                - self._train_start_time
                + self._train_time
                + self._val_time
            },
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        return res

    def validation_step(self, batch, batch_index):
        res = self._evaluate(batch, batch_index)
        self.validation_step_outputs.append(
            {
                **res,
                **{
                    "size": torch.tensor(
                        len(batch[self.batch_key]), device=self.run_device
                    )
                },
            }
        )
        return {
            **res,
            **{
                "size": torch.tensor(len(batch[self.batch_key]), device=self.run_device)
            },
        }

    def test_step(self, batch, batch_index):
        return self.validation_step(batch, batch_index)

    def get_metrics(self):
        items = super().get_metrics()
        items.pop("v_num", None)
        return items

    def configure_optimizers(self):
        if self.config["Optimiser"] == "SGD":
            return torch.optim.SGD(self.parameters(), lr=self.config["lr"])

        elif self.config["Optimiser"] == "ADAM":
            return torch.optim.Adam(
                self.parameters(), lr=self.config["lr"], betas=(0.9, 0.999), eps=1e-08
            )

        else:
            raise Exception("Optimiser not defined")
