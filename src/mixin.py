import abc
from copy import deepcopy
import torch
import torch.nn as nn
from tqdm import tqdm
from config import cfg
import os
import sys


class FeatureExtractor(abc.ABC):
    dim: int
    @abc.abstractmethod
    def fit(self, loader): pass
    @abc.abstractmethod
    def encode(self, images): pass


class FineTuneMixin:
    def _make_val_split(self, train_subset, val_ratio=1-cfg.splits.training_ratio):
        idx = train_subset.indices if hasattr(train_subset, "indices") else list(range(len(train_subset)))
        split = int(len(idx) * (1 - val_ratio))
        return (
            torch.utils.data.Subset(train_subset.dataset, idx[:split]),
            torch.utils.data.Subset(train_subset.dataset, idx[split:])
        )

    def _train_one_epoch(self, loader, criterion, optimizer, scheduler):
        self.backbone.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y, _ in tqdm(loader, disable=not cfg.verbose, leave=False):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = self.backbone(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()

            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
        return running_loss / total, correct / total

    @torch.no_grad()
    def _validate(self, loader, criterion):
        self.backbone.eval()
        vl, vc, vt = 0.0, 0, 0
        for x, y, _ in loader:
            x, y = x.cuda(), y.cuda()
            out = self.backbone(x)
            loss = criterion(out, y)
            vl += loss.item() * x.size(0)
            vc += (out.argmax(1) == y).sum().item()
            vt += x.size(0)
        return vl / vt, vc / vt

    def _fine_tune(self, train_loader, *, max_epochs=cfg.training.epochs, patience=3,
                   lr_head=cfg.training.lr_head, lr_base=cfg.training.lr_backbone, weight_decay=1e-4):
        train_ds, val_ds = self._make_val_split(train_loader.dataset, 0.1)
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=train_loader.batch_size, shuffle=True,
            num_workers=cfg.dataset.num_workers, drop_last=True)
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=train_loader.batch_size, shuffle=False,
            num_workers=cfg.dataset.num_workers)

        optim = torch.optim.AdamW(
            [
                {"params": self.head_params, "lr": lr_head, "weight_decay": weight_decay},
                {"params": self.base_params, "lr": lr_base, "weight_decay": weight_decay}
            ]
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs*len(train_dl))
        criterion = getattr(self, "loss_fn", nn.CrossEntropyLoss())
        best_wts, best_acc, wait = deepcopy(self.backbone.state_dict()), 0.0, 0

        for epoch in range(1, max_epochs + 1):
            tl, ta = self._train_one_epoch(train_dl, criterion, optim, sched)
            vl, va = self._validate(val_dl, criterion)
            print(f"Epoch {epoch:02d}: train loss={tl:.4f} acc={ta:.3f} | "
                  f"val loss={vl:.4f} acc={va:.3f}")
            if va > best_acc + 1e-4:
                best_acc, best_wts, wait = va, deepcopy(self.backbone.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered")
                    break

        self.backbone.load_state_dict(best_wts)
        print(f"Best val acc={best_acc:.3f} (epoch {epoch-wait})")