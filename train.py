# train.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from lifelines.utils import concordance_index
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, survival_type='cox'):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()
        self.survival_type = survival_type.lower()

    def forward(self, logits, y_auc, y_time, y_event):
        bce_loss = self.bce(logits, y_auc)

        if self.survival_type == 'cox':
            # 展开所有样本和事件维度，变成一维向量
            risk_pred = logits.view(-1)       # [B*947]
            y_time_flat = y_time.view(-1)     # [B*947]
            y_event_flat = y_event.view(-1)   # [B*947]
            survival_loss = self._cox_loss(risk_pred, y_time_flat, y_event_flat)
        elif self.survival_type == 'nll':
            risk_pred = logits.view(-1)
            y_time_flat = y_time.view(-1)
            y_event_flat = y_event.view(-1)
            survival_loss = self._nll_loss(risk_pred, y_time_flat, y_event_flat)
        else:
            raise ValueError(f"Unsupported survival_type: {self.survival_type}")

        return self.alpha * bce_loss + self.beta * survival_loss

    def _cox_loss(self, risk_pred, y_time, y_event, eps=1e-7):
        """
        Cox partial likelihood negative log loss
        Inputs:
            risk_pred: [N] 一维风险分数
            y_time: [N] 生存时间
            y_event: [N] 事件指示
        """
        idx = torch.argsort(-y_time)
        y_time = y_time[idx]
        y_event = y_event[idx]
        risk_pred = risk_pred[idx]

        log_cumsum = torch.logcumsumexp(risk_pred, dim=0)
        diff = risk_pred - log_cumsum
        loss = -torch.sum(diff * y_event) / (y_event.sum() + eps)
        return loss

    def _nll_loss(self, risk_pred, y_time, y_event):
        risk_pred = torch.clamp(risk_pred, min=1e-5)
        loss = F.mse_loss(risk_pred, y_time)  # 可改为更合理的时间回归损失
        return loss

from tqdm import tqdm
import torch

def train_one_epoch(model, loader, optimizer, Combined_Loss, device, epoch=None, total_epochs=None):
    model.train()
    total_loss = 0.
    num_samples = 0

    # 设置进度条显示的前缀，例如 "Epoch [1/50]"
    prefix = f"Epoch [{epoch}/{total_epochs}]" if epoch is not None and total_epochs is not None else "Training"

    pbar = tqdm(loader, desc=prefix, leave=True, dynamic_ncols=True)

    for batch in pbar:
        # 全部输入转到 device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        y = batch['y_auc'].to(device)
        y_time = batch['y_time'].to(device)
        y_event = batch['y_event'].to(device)

        logits = model(
            batch['demo'], batch['physical'], batch['biomarkers'], batch['lifestyle'],
            batch['mental'], batch['environmental'], batch['genetic'], batch['other'],
            batch['x_phecode'], batch['x_pheno'], batch['x_opcs4'], batch['x_drug']
        )

        # loss 计算 & 更新
        loss = Combined_Loss(logits, y, y_time, y_event)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累积 loss
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        # 更新 tqdm 显示
        avg_loss = total_loss / num_samples
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"})

    return total_loss / num_samples

from tqdm import tqdm
import torch
import numpy as np

def eval_model(model, loader, device, if_Test=False, save_path=None, epoch=None, total_epochs=None):
    model.eval()
    ys, probs, y_times, y_events = [], [], [], []
    logits_list, id_list = [], []

    prefix = f"Validation [{epoch}/{total_epochs}]" if epoch is not None else "Evaluating"
    pbar = tqdm(loader, desc=prefix, leave=True, dynamic_ncols=True)

    with torch.no_grad():
        total_samples = 0
        for batch in pbar:
            # 所有 tensor 放到 device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            y = batch['y_auc']
            y_time = batch['y_time']
            y_event = batch['y_event']

            logits = model(
                batch['demo'], batch['physical'], batch['biomarkers'], batch['lifestyle'],
                batch['mental'], batch['environmental'], batch['genetic'], batch['other'],
                batch['x_phecode'], batch['x_pheno'], batch['x_opcs4'], batch['x_drug']
            )

            prob = torch.sigmoid(logits)

            ys.append(y)
            probs.append(prob)
            y_times.append(y_time)
            y_events.append(y_event)

            if if_Test:
                logits_list.append(logits.cpu())  # 直接转 CPU
                # 转 CPU 并保存 id
                if torch.is_tensor(batch['id']):
                    id_list.extend(batch['id'].detach().cpu().tolist())
                else:
                    id_list.extend([
                        x.detach().cpu().item() if torch.is_tensor(x) else x
                        for x in batch['id']
                    ])

            # 更新进度条
            batch_size = y.size(0)
            total_samples += batch_size
            pbar.set_postfix({"batch_size": batch_size, "seen": total_samples})

    # 合并成大 tensor
    ys = torch.cat(ys, dim=0)
    probs = torch.cat(probs, dim=0)
    y_times = torch.cat(y_times, dim=0)
    y_events = torch.cat(y_events, dim=0)

    # 保存文件
    if if_Test and save_path is not None:
        logits_all = torch.cat(logits_list, dim=0).numpy()
        ids = np.array(id_list)
        np.savez_compressed(
            save_path,
            ids=ids,
            logits=logits_all,
            probs=probs.cpu().numpy(),
            ys=ys.cpu().numpy(),
            y_times=y_times.cpu().numpy(),
            y_events=y_events.cpu().numpy()
        )
        print(f"===> ✅ 已保存测试集数据到 {save_path}")

    return ys, probs, y_times, y_events


import torch
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision

def phecode_risk_metrics_torch(
    y_true, 
    y_prob, 
    y_time, 
    y_event, 
    device=None, 
    label_mask=None, 
    batch_size=2000, 
    verbose=False
):
    """
    计算 ROC-AUC, AUPRC, C-index（GPU 优化版，避免 OOM + NaN）
    - verbose=True  : 计算所有指标
    - verbose=False : 只计算 roc_auc_micro
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 转成 tensor & 送到 GPU
    y_true = torch.as_tensor(y_true, dtype=torch.int, device=device)
    y_prob = torch.as_tensor(y_prob, dtype=torch.float32, device=device)
    y_time = torch.as_tensor(y_time, dtype=torch.float32, device=device)
    y_event = torch.as_tensor(y_event, dtype=torch.float32, device=device)

    # 按 label_mask 过滤
    if label_mask is not None:
        label_mask = torch.as_tensor(label_mask, dtype=torch.bool, device=device)
        y_true = y_true[:, label_mask]
        y_prob = y_prob[:, label_mask]
        y_time = y_time[:, label_mask]
        y_event = y_event[:, label_mask]

    # 过滤全 0 / 全 1 标签
    valid_labels = (y_true.sum(dim=0) > 0) & (y_true.sum(dim=0) < y_true.shape[0])
    y_true = y_true[:, valid_labels]
    y_prob = y_prob[:, valid_labels]
    y_time = y_time[:, valid_labels]
    y_event = y_event[:, valid_labels]
    num_labels = y_true.shape[1]

    # 如果没有有效标签，直接返回 NaN
    if num_labels == 0:
        return {k: float('nan') for k in [
            "roc_auc_micro","auprc_micro","roc_auc_macro","auprc_macro",
            "roc_auc_weighted","auprc_weighted","c_index_mean","c_index_median"
        ]}

    metrics = {}

    # ===== 1. ROC-AUC & AUPRC =====
    try:
        roc_auc_micro = MultilabelAUROC(num_labels=num_labels, average="micro").to(device)(y_prob, y_true)
        metrics["roc_auc_micro"] = roc_auc_micro.item()

        if verbose:
            auprc_micro = MultilabelAveragePrecision(num_labels=num_labels, average="micro").to(device)(y_prob, y_true)
            roc_auc_macro = MultilabelAUROC(num_labels=num_labels, average="macro").to(device)(y_prob, y_true)
            auprc_macro = MultilabelAveragePrecision(num_labels=num_labels, average="macro").to(device)(y_prob, y_true)
            roc_auc_weighted = MultilabelAUROC(num_labels=num_labels, average="weighted").to(device)(y_prob, y_true)
            auprc_weighted = MultilabelAveragePrecision(num_labels=num_labels, average="weighted").to(device)(y_prob, y_true)

            metrics["auprc_micro"] = auprc_micro.item()
            metrics["roc_auc_macro"] = roc_auc_macro.item()
            metrics["auprc_macro"] = auprc_macro.item()
            metrics["roc_auc_weighted"] = roc_auc_weighted.item()
            metrics["auprc_weighted"] = auprc_weighted.item()
        else:
            # 只输出 roc_auc_micro，填充其他指标为 NaN
            for k in ["auprc_micro","roc_auc_macro","auprc_macro","roc_auc_weighted","auprc_weighted"]:
                metrics[k] = float('nan')

    except Exception:
        for k in ["roc_auc_micro","auprc_micro","roc_auc_macro","auprc_macro","roc_auc_weighted","auprc_weighted"]:
            metrics[k] = float('nan')

    # ===== 2. 分批计算 C-index =====
    def batched_cindex(times, events, preds, batch_size=2000):
        n = len(times)
        concordant = torch.tensor(0.0, device=times.device)
        total = torch.tensor(0.0, device=times.device)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            t_batch = times[start:end]
            e_batch = events[start:end]
            p_batch = preds[start:end]
            time_diff = t_batch.unsqueeze(1) < times.unsqueeze(0)
            risk_diff = p_batch.unsqueeze(1) > preds.unsqueeze(0)
            valid_pairs = time_diff & e_batch.unsqueeze(1).bool()
            concordant += (risk_diff & valid_pairs).sum()
            total += valid_pairs.sum()
        return (concordant / total).item() if total > 0 else float('nan')

    if verbose:
        cindices = []
        for i in range(num_labels):
            mask = ~torch.isnan(y_time[:, i]) & ~torch.isnan(y_prob[:, i]) & ~torch.isnan(y_event[:, i])
            if mask.sum() < 2:
                continue
            cidx = batched_cindex(y_time[mask, i], y_event[mask, i], y_prob[mask, i], batch_size=batch_size)
            if not torch.isnan(torch.tensor(cidx)):
                cindices.append(cidx)

        if cindices:
            metrics["c_index_mean"] = float(torch.tensor(cindices, device=device).mean().item())
            metrics["c_index_median"] = float(torch.tensor(cindices, device=device).median().item())
        else:
            metrics["c_index_mean"] = float('nan')
            metrics["c_index_median"] = float('nan')
    else:
        metrics["c_index_mean"] = float('nan')
        metrics["c_index_median"] = float('nan')

    return metrics


class EarlyStopping:
    def __init__(self, patience=10, mode="min", delta=1e-4):
        """
        Args:
            patience (int): 超过多少个 epoch 没有提升就早停
            mode (str): "min" 表示越小越好 (如 val_loss)，"max" 表示越大越好 (如 AUC)
            delta (float): 认为有提升的最小变化值
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        else:
            if self.mode == "min":
                if current_score < self.best_score - self.delta:
                    self.best_score = current_score
                    self.counter = 0
                else:
                    self.counter += 1
            elif self.mode == "max":
                if current_score > self.best_score + self.delta:
                    self.best_score = current_score
                    self.counter = 0
                else:
                    self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
