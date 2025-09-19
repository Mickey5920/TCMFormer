# main.py
from datetime import datetime
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset,random_split,ConcatDataset,RandomSampler
from pandas.api.types import is_numeric_dtype
import csv
import gc
import open_clip
import os
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from train import train_one_epoch,eval_model,CombinedLoss,phecode_risk_metrics_torch, EarlyStopping
# from model import CapsuleTransformerMultiDiseaseRiskModel,build_model
from model import CapsuleTransformerMultiDiseaseRiskModel
from Dataloader import group_map,label_encode_df,vectorize_all_series_resume,MultiModalRiskDataset,get_data_loaders,merge_pt_batches
from sklearn.decomposition import PCA
import random
import math
def set_seed(seed=42):
    random.seed(seed)                  # Python 随机数
    np.random.seed(seed)               # NumPy 随机数
    torch.manual_seed(seed)             # PyTorch CPU 随机数
    torch.cuda.manual_seed(seed)        # 当前 GPU
    torch.cuda.manual_seed_all(seed)    # 所有 GPU（多卡）
    
    # cudnn 固定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"随机种子已固定为 {seed}")

def parse_args():
    parser = argparse.ArgumentParser(description='Multi_Disease Prediction Model Training')
    parser.add_argument('--All_use_data',type=str,default='./All_use_data/All_Modalities_Aligned_test.csv',help='All Use Data Path')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dataset_path',type=str,default='/data/multi_disease_risk_dataset.pt',help='Path to the dataset')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(2025)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if os.path.exists(args.dataset_path):
        print(f"-----------------Loads existing dataset: {args.dataset_path}-----------------")
        full_dataset = torch.load(args.dataset_path, weights_only=False)

    else:
        print("-----------------Rebuild Dataset and save-----------------")

        print("-----------------Read CSV data-----------------")
        if not os.path.exists(args.All_use_data):
            # 1. 读取数据
            df_base = pd.read_csv(
                r'F:\Code\Disease_Prediction\Data-collection\15-Combined_Base_Information_Phenotypes.csv',
                dtype={'Participant ID': str})
            df_icd = pd.read_csv(
                r'F:\Code\Disease_Prediction\Data-collection\12-ICD_Before_After_Baseline_Data_Date_Phecode.csv',
                dtype={'Participant ID': str})
            df_gp = pd.read_csv(
                r'F:\Code\Disease_Prediction\Data-collection\13-GP_Medications_Before_After_Baseline_Data_Date.csv',
                dtype={'Participant ID': str})
            df_op = pd.read_csv(
                r'F:\Code\Disease_Prediction\Data-collection\14-Operative_Procedures_Before_After_Baseline_Data_Date.csv',
                dtype={'Participant ID': str})
            df_genetic = pd.read_csv(
                r'F:\Code\Disease_Prediction\Data-collection\17-UKB_Genetic_PCs_and_PRS.csv',
            )
            df_base['Participant ID'] = df_base['Participant ID'].astype(str).str.strip()
            df_icd['Participant ID'] = df_icd['Participant ID'].astype(str).str.strip()
            df_gp['Participant ID'] = df_gp['Participant ID'].astype(str).str.strip()
            df_op['Participant ID'] = df_op['Participant ID'].astype(str).str.strip()
            df_genetic['Participant ID'] = df_genetic['Participant ID'].astype(str).str.strip()
            # 2. 补全缺失值（只针对缺失率<10%）
            for col in df_base.columns:
                if col == 'Participant ID':
                    continue

                # 判断列是否连续变量
                _tmp = pd.to_numeric(df_base[col], errors='coerce')
                is_continuous = (_tmp.notna().any()) and (
                        (_tmp.astype(float).dropna() % 1 != 0).any() or _tmp.max() - _tmp.min() > 10)

                na_count = df_base[col].isna().sum()
                na_ratio = na_count / len(df_base)

                if is_continuous:
                    # 连续变量：缺失率小于10%用中位数补全
                    if 0 < na_ratio < 0.35:
                        median_val = _tmp.median()
                        df_base[col].fillna(median_val, inplace=True)
                    # 否则不处理，保留NaN
                else:
                    # 分类型变量：全部缺失都置为0
                    df_base[col].fillna(0, inplace=True)
            print(f'df_base：共{df_base.shape[0]}人，{df_base.shape[1]}列')
            df_base = df_base.dropna()
            print(f'Dropna df_base：共{df_base.shape[0]}人，{df_base.shape[1]}列')
            df_all = pd.merge(df_base, df_genetic, on='Participant ID', how='left')
            df_base = df_base.dropna()
            print(f'merge genetic dropna：共{df_base.shape[0]}人，{df_base.shape[1]}列')
            # 3. 合并
            df_all = pd.merge(df_all, df_icd, on='Participant ID', how='left')
            df_all = pd.merge(df_all, df_gp, on='Participant ID', how='left')
            df_all = pd.merge(df_all, df_op, on='Participant ID', how='left')

            print(f'Dropna之后，最终多模态表已保存，共{df_all.shape[0]}人，{df_all.shape[1]}列')
            # input("请按任意键继续...")
            # 4. 保存
            df_all.to_csv(args.All_use_data, index=False, encoding='utf-8-sig')
        else:
            print(f'-----------------read {args.All_use_data}-----------------')
            df_all = pd.read_csv(args.All_use_data, low_memory=False)
            print(f'多模态表已读取，共{df_all.shape[0]}人，{df_all.shape[1]}列')
            Genetic_cols = ([f'Genetic principal components | Array {i}' for i in range(1, 21)]
                        + [f'PRS genetic principal components | Array {i}' for i in range(0, 4)])
            # 只保留这24列全部非缺失的行
            df_all_clean = df_all.dropna(subset=Genetic_cols, how='any')
            print(f"原始人数：{df_all.shape[0]}，去除这24列有缺失后剩余：{df_all_clean.shape[0]}")
            # df_all_clean = df_all_clean.head(10000)
            # sample_length = len(df_all_clean)

        print("-----------------Data Process-----------------")
        # 2. 增加基因主成分和PRS到分组
        genetic_pcs = [f'Genetic principal components | Array {i}' for i in range(1, 21)]
        prs_pcs = [f'PRS genetic principal components | Array {i}' for i in range(0, 4)]

        for col in genetic_pcs + prs_pcs:
            group_map[col] = 'Genetic Factors'
        # 所有未分组变量自动归入 Other
        # 自动分组与导出
        # 1. 分组拆分
        grouped_dfs = {}
        for col in df_all_clean.columns:
            group = group_map.get(col, 'Other')
            grouped_dfs.setdefault(group, []).append(col)

        # 2. 按分组生成各自DataFrame
        for group, col_list in grouped_dfs.items():
            grouped_dfs[group] = df_all_clean[col_list]

        # 3. 示例：对每组数据做自定义处理
        for group, df_group in grouped_dfs.items():
            print(f'== {group} ==')
            print(df_group.shape)
        Diagnosis = grouped_dfs['Diagnosis (Medical Records)']
        Medications = grouped_dfs['Medications (Medical Records)']
        Operative_Procedures = grouped_dfs['Operative Procedures (Medical Records)']

        Demographics = grouped_dfs['Demographics']
        Physical_Measures = grouped_dfs['Physical Measures']
        Biomarkers = grouped_dfs['Biomarkers']
        Lifestyle_Factors = grouped_dfs['Lifestyle Factors']
        Mental_Health_and_Psychosocial = grouped_dfs['Mental Health and Psychosocial Factors']
        Environmental_Exposures = grouped_dfs['Environmental Exposures']
        Genetic_Factors = grouped_dfs['Genetic Factors']
        Other = grouped_dfs['Other']

        Participant_IDs = df_all_clean['Participant ID'].copy() # 1000086

        Demo_le = label_encode_df(Demographics)
        Physical_le = label_encode_df(Physical_Measures)
        Biomarkers_le = label_encode_df(Biomarkers)
        Lifestyle_le = label_encode_df(Lifestyle_Factors)
        Mental_le = label_encode_df(Mental_Health_and_Psychosocial)
        Environmental_le = label_encode_df(Environmental_Exposures)
        Genetic_le = label_encode_df(Genetic_Factors)
        Other_le = label_encode_df(Other)

        print(f"-----------------OpenCLIP Init-----------------")
        # --- 1. 初始化OpenCLIP模型与分词器 ---
        model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
        model_clip = model_clip.to(device)
        model_clip.eval()
        tokenizer = open_clip.get_tokenizer('ViT-B-16')

        vectorize_all_series_resume(
            model_clip, tokenizer,
            Diagnosis['Before_Baseline-PheCode'], Diagnosis['Before_Baseline-ICD_Days_To_Baseline'],
            Diagnosis['Before_Baseline-Phenotype'], Diagnosis['Before_Baseline-Phenotype'],
            Medications['Before_Baseline-Drug'], Medications['Before_Baseline-Drug_Days_To_Baseline'],
            Operative_Procedures['Before_Baseline-OPCS4'],
            Operative_Procedures['Before_Baseline-OPCS4_Days_To_Baseline'],
            device='cuda', max_events=5, embed_dim=512, pca_dim=62, batch_size=128,
            save_dir="temp_batches"
        )

        save_dir = "temp_batches"

        X_before_phecode = merge_pt_batches(save_dir, "phecode", os.path.join(save_dir, "phecode_vec.pt"))
        X_before_phenotypes = merge_pt_batches(save_dir, "phenotypes", os.path.join(save_dir, "phenotypes_vec.pt"))
        X_before_opcs4 = merge_pt_batches(save_dir, "OPCS4", os.path.join(save_dir, "OPCS4_vec.pt"))
        X_before_drug = merge_pt_batches(save_dir, "drug", os.path.join(save_dir, "drug_vec.pt"))

        y_auc = np.stack(Diagnosis['y_auc'].apply(lambda x: np.array(list(map(int, x.split(','))))).values)
        Y_time = np.stack(Diagnosis['Y_time'].apply(lambda x: np.array(list(map(int, x.split(','))))).values)
        Y_event = np.stack(Diagnosis['Y_event'].apply(lambda x: np.array(list(map(int, x.split(','))))).values)


        def print_type_and_shape(var, name):
            if isinstance(var, pd.DataFrame) or isinstance(var, pd.Series):
                nan_count = var.isna().sum().sum() if isinstance(var, pd.DataFrame) else var.isna().sum()
                print(f"{name}: pandas object, NaN count = {nan_count}")
            elif isinstance(var, np.ndarray):
                nan_count = np.isnan(var).sum()
                print(f"{name}: numpy.ndarray, NaN count = {nan_count}")
            elif isinstance(var, torch.Tensor):
                nan_count = torch.isnan(var).sum().item()
                print(f"{name}: torch.Tensor, NaN count = {nan_count}")
            else:
                print(f"{name}: type {type(var)} not supported for NaN check")

        print_type_and_shape(Participant_IDs, "Participant_IDs")
        print_type_and_shape(Demo_le, "Demo_le")
        print_type_and_shape(Physical_le, "Physical_le")
        print_type_and_shape(Biomarkers_le, "Biomarkers_le")
        print_type_and_shape(Lifestyle_le, "Lifestyle_le")
        print_type_and_shape(Mental_le, "Mental_le")
        print_type_and_shape(Environmental_le, "Environmental_le")
        print_type_and_shape(Genetic_le, "Genetic_le")
        print_type_and_shape(Other_le, "Other_le")
        print_type_and_shape(X_before_phecode, "X_before_phecode")
        print_type_and_shape(X_before_phenotypes, "X_before_phenotypes")
        print_type_and_shape(X_before_opcs4, "X_before_opcs4")
        print_type_and_shape(X_before_drug, "X_before_drug")
        print_type_and_shape(y_auc, "y_auc")
        print_type_and_shape(Y_time, "Y_time")
        print_type_and_shape(Y_event, "Y_event")

        dataset = MultiModalRiskDataset(
            Demo_le, Physical_le, Biomarkers_le, Lifestyle_le, Mental_le,
            Environmental_le, Genetic_le, Other_le,
            X_before_phecode, X_before_phenotypes, X_before_opcs4, X_before_drug,
            y_auc, Y_time, Y_event, Participant_IDs
        )
        print('-----------------Save dataset to disk-----------------')
        torch.save(dataset, args.dataset_path)

        # 加载检查
        full_dataset = torch.load(args.dataset_path, weights_only=False)
        print("Total samples in dataset:", len(full_dataset))
        print("Sample[0] keys:", full_dataset[0].keys())

    # 划分数据
    print('-----------------Split Data-----------------')
    train_loader, val_loader, test_loader, all_loader = get_data_loaders(full_dataset, train_size=0.8, preload=True, batch_size=args.batch_size)

    print(r'-----------------Model Init-----------------')
    tab_dims = [7, 24, 62, 58, 22, 8, 24, 69]  # 依次对应各结构化变量列数
    event_feat_dim = 64  # 事件序列输入维度 (如5,64)
    output_dim = full_dataset[0]['y_auc'].shape[0]  # 标签数，如947
    hidden_dim = 256


    model = CapsuleTransformerMultiDiseaseRiskModel(
            tab_dims=tab_dims,
            event_feat_dim=event_feat_dim,
            out_dim=output_dim,
            hidden_dim=hidden_dim
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    Combined_Loss = CombinedLoss(alpha=1.0, beta=1.0, survival_type='cox')

    # ---- 动态学习率调节器 ----
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',              # 监控的指标越大越好
        factor=0.5,              # 学习率降低倍率
        patience=5,              # 连续多少个 epoch 无提升才降低学习率
        min_lr=1e-7,              # 学习率下限
        verbose=True             # 输出学习率变化日志
    )

    early_stopping = EarlyStopping(patience=10, mode="max")  # 比如监控 AUC

    print('-----------------Training-----------------')
    formatted_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    best_val_auc = 0
    for epoch in tqdm(range(args.num_epochs)):
        train_loss = train_one_epoch(model, train_loader, optimizer, Combined_Loss, device)

        print(f"[Epoch {epoch}] TrainLoss: {train_loss:.4f}")
        # ---- 验证集评价 ----
        yv, probv, y_timev, y_eventv = eval_model(model, val_loader, device)

        # ---- 计算验证集指标 ----
        label_mask = (yv.sum(axis=0) >= 1) if yv.shape[0] >= 1 else np.ones(yv.shape[1], dtype=bool)
        if epoch%10==0:
            val_metrics = phecode_risk_metrics_torch(yv, probv, y_timev, y_eventv, device, label_mask=label_mask, verbose=True)
        else:
            val_metrics = phecode_risk_metrics_torch(yv, probv, y_timev, y_eventv, device, label_mask=label_mask, verbose=False)
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {str(v)[:120]}...")

        # ---- 更新学习率 ----
        # scheduler.step()
        # ---- 根据验证集 AUC 调整学习率 ----
        scheduler.step(val_metrics['roc_auc_micro'])
        # === 检查早停 ===
        if early_stopping(val_metrics['roc_auc_micro']):
            print(f"-----------------Early stopping at epoch {epoch}-----------------")
            break
        # --- 保存最优模型 ---
        if val_metrics['roc_auc_micro'] > best_val_auc:
            print(f"Best model at epoch {epoch} with val auc {val_metrics['roc_auc_micro']:.4f}")
            torch.save(model.state_dict(), f'checkpoint/{formatted_date}_best_model.pth')
            best_val_auc = val_metrics['roc_auc_micro']

    # 7. 测试集评价
    print('-----------------Test Dataset Evaluation-----------------')
    model.load_state_dict(torch.load(f'checkpoint/{formatted_date}_best_model.pth', weights_only=False))
    yv, probv, y_timev, y_eventv = eval_model(model, test_loader, device, if_Test=True,save_path=f"/data/{formatted_date}_{args.compare_model}_test_preds.npz")
    
    # ---- 计算测试集指标 ----
    label_mask = (yv.sum(axis=0) >= 1) if yv.shape[0] >= 1 else np.ones(yv.shape[1], dtype=bool)
    val_metrics = phecode_risk_metrics_torch(yv, probv, y_timev, y_eventv, device, label_mask=label_mask, verbose=True)
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {str(v)[:120]}...")

    print('-----------------All Dataset Evaluation-----------------')
    yv, probv, y_timev, y_eventv = eval_model(model, all_loader, device, if_Test=True, save_path=f"/data/{formatted_date}_{args.compare_model}_All_preds.npz")
    # label_mask = (yv.sum(axis=0) >= 1) if yv.shape[0] >= 1 else np.ones(yv.shape[1], dtype=bool)
    # val_metrics = phecode_risk_metrics_torch(yv, probv, y_timev, y_eventv, device, label_mask=label_mask, verbose=True)
    # for k, v in val_metrics.items():
    #     print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {str(v)[:120]}...")
if __name__ == '__main__':
    main()