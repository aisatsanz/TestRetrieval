from pytorch_metric_learning import losses, miners, samplers
import torch.nn as nn
import torch
import timm
from torchvision import transforms
import os
import torch.nn.functional as F
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
import faiss
from .mixin import FineTuneMixin, FeatureExtractor
from config import cfg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNetExtractor(FineTuneMixin, FeatureExtractor):
    def __init__(self, num_classes=5):
        self.backbone = timm.create_model('resnet50', pretrained=True, drop_rate=0.2)
        self.backbone.reset_classifier(num_classes)
        self.dim = self.backbone.num_features

        classifier = self.backbone.get_classifier()
        self.head_params = list(classifier.parameters())
        self.base_params = [p for p in self.backbone.parameters() if id(p) not in set(map(id, self.head_params))]

    def fit(self, train_loader):
        print("Fine tune ResNet50")
        self.backbone.cuda()
        self._fine_tune(train_loader,
                        max_epochs=cfg.training.epochs,
                        patience=3,
                        lr_head=cfg.training.lr_head,
                        lr_base=cfg.training.lr_backbone)

    @staticmethod
    def _pool(feats):
        """
        B*C*H*W  ->  B*C или оставляет если уже
        """
        if feats.ndim == 4:
            feats = feats.mean(dim=(-1, -2))
        elif feats.ndim == 3:
            feats = feats.squeeze(-1)
        return torch.nn.functional.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode(self, images):
        if isinstance(images, torch.Tensor):
            feats = self.backbone.forward_features(images.to(DEVICE))
            feats = self._pool(feats)
            return feats.cpu().numpy().astype('float32')


class EfficientNetExtractor(FineTuneMixin, FeatureExtractor):
    def __init__(self, num_classes=5):
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            drop_rate=0.2
        )
        self.backbone.reset_classifier(num_classes)
        self.dim = self.backbone.num_features
        classifier = self.backbone.get_classifier()
        self.head_params = list(classifier.parameters())
        self.base_params = [p for p in self.backbone.parameters()
                            if id(p) not in set(map(id, self.head_params))]

    def fit(self, train_loader):
        print("Fine tune EfficientNet-B0")
        self.backbone.cuda()
        self._fine_tune(train_loader,
                        max_epochs=cfg.training.epochs,
                        patience=3,
                        lr_head=cfg.training.lr_head,
                        lr_base=cfg.training.lr_backbone)
    @staticmethod
    def _pool(feats):
        """
        B*C*H*W  ->  B*C или оставляет если уже
        """
        if feats.ndim == 4:
            feats = feats.mean(dim=(-1, -2))
        elif feats.ndim == 3:
            feats = feats.squeeze(-1)
        return torch.nn.functional.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode(self, images):
        if isinstance(images, torch.Tensor):
            images = images.to(DEVICE)
            feats = self.backbone.forward_features(images)
            feats = self._pool(feats)
            return feats.cpu().numpy().astype('float32')
        
class CLIPHFExtractor(FeatureExtractor):
    """
    CLIP ViT-B/32 по умолчанию без fine tune
    """
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model     = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.dim = self.model.config.projection_dim
        self.to_pil = transforms.ToPILImage()

    def fit(self, *_): pass

    @torch.no_grad()
    def encode(self, images):
        if isinstance(images, torch.Tensor):
            pil = [self.to_pil(img.cpu()) for img in images]
        else:
            pil = [Image.open(p).convert("RGB") for p in images]
        inputs = self.processor(images=pil, return_tensors="pt", padding=True).to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = F.normalize(feats, p=2, dim=-1)
        return feats.cpu().numpy().astype("float32")
    

class MetricExtractor(nn.Module):
    """
    ResNet-18 backbone + Linear(256) + TripletLoss
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet18", pretrained=True, num_classes=0, drop_rate=0.2
        )
        self.embed = nn.Linear(self.backbone.num_features, embed_dim)
        self.dim   = embed_dim

        self.loss_fn = losses.TripletMarginLoss(margin=0.2)
        self.miner   = miners.TripletMarginMiner(
            margin=0.2, type_of_triplets="semihard"
        )

    def fit(
        self,
        dl: DataLoader,
        *,
        epochs: int = cfg.training.epochs,
        patience: int = 4,
        lr_head: float = cfg.training.lr_head,
        lr_base: float = cfg.training.lr_backbone,
        weight_decay: float = 1e-4,
    ):
        idx = dl.dataset.indices if hasattr(dl.dataset, "indices") else range(len(dl.dataset))
        n_val = int(0.1 * len(idx))
        train_idx, val_idx = idx[:-n_val], idx[-n_val:]

        base_ds = dl.dataset.dataset if hasattr(dl.dataset, "dataset") else dl.dataset
        train_ds = Subset(base_ds, train_idx)
        val_ds   = Subset(base_ds, val_idx)

        train_dl = DataLoader(
            train_ds, batch_size=dl.batch_size, shuffle=True,
            num_workers=cfg.dataset.num_workers, drop_last=True
        )
        val_dl = DataLoader(
            val_ds, batch_size=dl.batch_size, shuffle=False,
            num_workers=cfg.dataset.num_workers
        )

        opt = torch.optim.AdamW(
            [
                {"params": self.embed.parameters(), "lr": lr_head, "weight_decay": weight_decay},
                {"params": self.backbone.parameters(), "lr": lr_base, "weight_decay": weight_decay},
            ]
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs * len(train_dl)
        )

        self.backbone.cuda(); self.embed.cuda()

        best_knn, wait = 0.0, 0
        best_state     = {k: v.clone() for k, v in self.state_dict().items()}

        for ep in range(1, epochs + 1):

            self.train()
            run_loss, seen = 0.0, 0
            for x, y, _ in tqdm(train_dl, leave=False, desc=f"E{ep:02d} train"):
                x, y = x.cuda(), y.cuda()
                opt.zero_grad()
                emb = F.normalize(self.embed(self.backbone(x)), dim=-1)
                hard = self.miner(emb, y)
                loss = self.loss_fn(emb, y, hard)
                loss.backward(); opt.step(); sched.step()
                run_loss += loss.item() * x.size(0)
                seen     += x.size(0)
            train_loss = run_loss / seen
            knn_acc = self._val_knn_acc(train_dl, val_dl, k=5)

            print(f"Epoch {ep:02d}: loss={train_loss:.4f} | val kNN@1={knn_acc:.3f}")

            if knn_acc > best_knn + 1e-4:
                best_knn, best_state, wait = knn_acc, \
                    {k: v.clone() for k, v in self.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping.")
                    break

        self.load_state_dict(best_state)
        print(f"Best val kNN@1 = {best_knn:.3f}")

    @torch.no_grad()
    def _val_knn_acc(self, train_dl: DataLoader, val_dl: DataLoader, *, k: int = 5) -> float:
        """
        Строит Flat-индекс из train-эмбеддингов и меряет top-1 точность на val
        """
        tr_vecs, tr_lbls = [], []
        for x, y, _ in train_dl:
            z = F.normalize(self.embed(self.backbone(x.cuda())), dim=-1)
            tr_vecs.append(z.cpu()); tr_lbls.append(y)
        tr_vecs = torch.cat(tr_vecs).numpy().astype("float32")
        tr_lbls = torch.cat(tr_lbls).numpy()

        faiss.normalize_L2(tr_vecs)
        index = faiss.IndexFlatIP(tr_vecs.shape[1])
        index.add(tr_vecs)

        correct, total = 0, 0
        for x, y, _ in val_dl:
            q = F.normalize(self.embed(self.backbone(x.cuda())), dim=-1).cpu().numpy()
            faiss.normalize_L2(q)
            _, I = index.search(q, k)                # (B,k)
            pred = tr_lbls[I[:, 0]]                  # ближайший сосед
            correct += np.sum(pred == y.numpy())
            total   += y.size(0)
        return correct / total

    @torch.no_grad()
    def encode(self, images):
        """Возвращает numpy (B,D)"""
        if isinstance(images, torch.Tensor):
            x = images
        else:
            x = torch.stack([clip_val_tf(Image.open(p).convert("RGB")) for p in images])
        x = x.to(DEVICE)

        z = F.normalize(self.embed(self.backbone(x)), dim=-1)
        return z.cpu().numpy().astype("float32")
    

class DINOv2Extractor(FeatureExtractor):
    def __init__(self, model_name: str = "vit_base_patch14_dinov2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.IMG = 518
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            img_size=self.IMG,          # задаём правильный размер тк дино на 518
        ).to(self.device).eval()

        self.dim = self.backbone.num_features

        self.tf = transforms.Compose([
            transforms.Resize(self.IMG, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.IMG),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def fit(self, *_): 
        pass

    @torch.no_grad()
    def encode(self, images):
        if isinstance(images, torch.Tensor):
            x = images
        else: 
            pil = [Image.open(p).convert("RGB") for p in images]
            x = torch.stack([self.tf(im) for im in pil])
        x = x.to(DEVICE)
        if isinstance(self.IMG, int):
            target_size = (self.IMG, self.IMG)
        else:                                         
            target_size = self.IMG

        if x.shape[-2:] != target_size:            
            x = F.interpolate(x, size=target_size,
                            mode="bicubic", align_corners=False)

        feats = self.backbone(x)                       # B * D
        feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype("float32")


class FastMetricExtractor(nn.Module):
    def __init__(self, embed_dim=512, n_classes=5):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        for name, p in self.backbone.named_parameters():
            if not name.startswith("layer4"):       
                p.requires_grad_(False)

        self.embed = nn.Linear(self.backbone.num_features, embed_dim)
        self.loss  = losses.ProxyAnchorLoss(n_classes, embed_dim).to(self.device)

    def forward(self, x):
        return self.backbone(x)


    def fit(self, dl: DataLoader, *, epochs=5, lr_backbone=1e-5, lr_head=1e-4):

        sampler = samplers.MPerClassSampler(dl.dataset, m=2,
                                            length_before_new_iter=len(dl.dataset))
        train_dl = DataLoader(dl.dataset,
                              batch_size = cfg.training.batch_size,
                              sampler    = sampler,
                              num_workers= cfg.dataset.num_workers,
                              drop_last  = True)


        opt = torch.optim.AdamW([
            {"params": self.embed.parameters(),                 "lr": lr_head},
            {"params": filter(lambda p: p.requires_grad,
                              self.backbone.parameters()),      "lr": lr_backbone},
            {"params": self.loss.parameters(),                  "lr": lr_head}
        ])

        self.to(self.device)
        best_map, best_state = 0., None

        for ep in range(1, epochs + 1):
            self.train()
            loop = tqdm(train_dl, total=len(train_dl),
                        desc=f"E{ep:02d} train", leave=False)

            for x, y, _ in loop:
                x, y = x.to(self.device), y.to(self.device)

                opt.zero_grad()
                emb  = F.normalize(self.embed(self.backbone(x)), dim=-1)
                loss = self.loss(emb, y)
                loss.backward()
                opt.step()

                loop.set_postfix(loss=f"{loss.item():.4f}")

    
            p5_val, map_val = self._quick_map(train_dl, k=5)
            print(f"E{ep:02d}: P@5={p5_val:.3f} | mAP@5={map_val:.3f}")

            if map_val > best_map:
                best_map, best_state = map_val, {k: v.clone()
                                                 for k, v in self.state_dict().items()}

        self.load_state_dict(best_state)
        print(f"✔ Best mAP={best_map:.3f}")


    @torch.no_grad()
    def _quick_map(self, dl, k: int = 5):
        vecs, lbls = [], []
        for x, y, _ in tqdm(dl, desc="encode val", leave=False):
            z = F.normalize(self.embed(self.backbone(x.to(self.device))), dim=-1).cpu()
            vecs.append(z); lbls.append(y)
        vecs = torch.cat(vecs).numpy().astype("float32")
        lbls = torch.cat(lbls).numpy()

        faiss.normalize_L2(vecs)
        index = faiss.IndexFlatIP(vecs.shape[1])
        if faiss.get_num_gpus():
            index = faiss.index_cpu_to_all_gpus(index)
        index.add(vecs)

        D, I = index.search(vecs, k)
        rel  = (lbls[I] == lbls[:, None]).astype(int)

        p_at_k = rel.sum(axis=1) / k
        mean_p = float(p_at_k.mean())

        precisions = rel.cumsum(1) / np.arange(1, k + 1)
        map_k = float((precisions * rel).sum() / rel.sum())

        return mean_p, map_k

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        x   = x.to(self.device, non_blocking=True)
        emb = F.normalize(self.embed(self.backbone(x)), dim=-1)
        return emb.cpu().numpy().astype("float32")