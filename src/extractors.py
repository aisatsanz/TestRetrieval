from pytorch_metric_learning import losses, miners
import torch.nn as nn
import torch
import timm
from torchvision import transforms
from PIL import Image
from tqdm import tqdm



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
            feats = self.backbone.forward_features(images.cuda())
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
            feats = self.backbone.forward_features(images.cuda())
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
            x = images.cuda()
        else:
            x = torch.stack([clip_val_tf(Image.open(p).convert("RGB")) for p in images]).cuda()

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
            x = images.to(self.device)
            if x.shape[-1] != self.IMG:
                x = F.interpolate(x, size=self.IMG, mode="bicubic", align_corners=False)
        else:
            x = torch.stack([self.tf(Image.open(p).convert("RGB")) for p in images]).to(self.device)

        feats = self.backbone(x)            # B * 768
        feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype("float32")

