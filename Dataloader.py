# Dataloader.py

import torch
import  random
mask = [0] * 7
from torch.utils.data import DataLoader, Dataset,random_split,IterableDataset,RandomSampler
import h5py
import pandas as pd
import ast
import open_clip
from sklearn.decomposition import IncrementalPCA
import numpy as np
import time
import threading
import queue
import os
from tqdm import tqdm
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelEncoder



def label_encode_df(df):
    df_le = df.copy()
    for col in df_le.columns:
        if df_le[col].dtype == object or isinstance(df_le[col].dtype, pd.StringDtype):
            le = LabelEncoder()
            df_le[col] = le.fit_transform(df_le[col].astype(str))
    return df_le

def collate_dict(batch):
    batch_dict = {}
    for key in batch[0]:
        values = [x[key] for x in batch]
        if isinstance(values[0], torch.Tensor):
            batch_dict[key] = torch.stack(values)
        elif isinstance(values[0], (int, float, np.integer, np.floating)):
            # 标量型：转为Tensor再堆叠
            batch_dict[key] = torch.tensor(values)
        elif isinstance(values[0], str):
            batch_dict[key] = values
        else:
            # numpy array（如某些情况），转Tensor
            try:
                batch_dict[key] = torch.stack([torch.as_tensor(v) for v in values])
            except Exception:
                batch_dict[key] = values  # fallback
    return batch_dict

def get_data_loaders(dataset, train_size=0.8, preload=False,batch_size=1):
    # Create dataset from .npz files in data_dir
    # Calculate the sizes for each split
    total_len = len(dataset)
    train_size = int(total_len * 0.8)
    val_size = int(total_len * 0.1)
    test_size = total_len - train_size - val_size  # 确保三者和为total_len

    # Split the dataset into train, validation, and test subsets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))

    # Create data loaders for each subset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_dict)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create a data loader for the entire dataset (all_loader)
    all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, all_loader
class MultiModalRiskDataset(Dataset):
    def __init__(
        self,
        demo, physical, biomarkers, lifestyle, mental, environmental, genetic, other,
        x_phecode, x_pheno, x_opcs4, x_drug,
        y_auc, y_time, y_event, ids=None
    ):
        self.demo = torch.tensor(demo.values, dtype=torch.float)
        self.physical = torch.tensor(physical.values, dtype=torch.float)
        self.biomarkers = torch.tensor(biomarkers.values, dtype=torch.float)
        self.lifestyle = torch.tensor(lifestyle.values, dtype=torch.float)
        self.mental = torch.tensor(mental.values, dtype=torch.float)
        self.environmental = torch.tensor(environmental.values, dtype=torch.float)
        self.genetic = torch.tensor(genetic.values, dtype=torch.float)
        self.other = torch.tensor(other.values, dtype=torch.float)
        self.x_phecode = x_phecode.float() # (N,5,64)
        self.x_pheno = x_pheno.float() # (N,5,64)
        self.x_opcs4 = x_opcs4.float() # (N,5,64)
        self.x_drug = x_drug.float() # (N,5,64)
        # self.x_phecode = torch.tensor(x_phecode, dtype=torch.float)
        # self.x_pheno = torch.tensor(x_pheno, dtype=torch.float)
        # self.x_opcs4 = torch.tensor(x_opcs4, dtype=torch.float)
        # self.x_drug = torch.tensor(x_drug, dtype=torch.float)
        self.y_auc = torch.tensor(y_auc, dtype=torch.float)             # (N,L)
        self.y_time = torch.tensor(y_time, dtype=torch.float)
        self.y_event = torch.tensor(y_event, dtype=torch.float)
        self.ids = list(ids.values) if ids is not None else None

    def __getitem__(self, idx):
        return {
            "demo": self.demo[idx],
            "physical": self.physical[idx],
            "biomarkers": self.biomarkers[idx],
            "lifestyle": self.lifestyle[idx],
            "mental": self.mental[idx],
            "environmental": self.environmental[idx],
            "genetic": self.genetic[idx],
            "other": self.other[idx],
            "x_phecode": self.x_phecode[idx],
            "x_pheno": self.x_pheno[idx],
            "x_opcs4": self.x_opcs4[idx],
            "x_drug": self.x_drug[idx],
            "y_auc": self.y_auc[idx],
            "y_time": self.y_time[idx],
            "y_event": self.y_event[idx],
            "id": self.ids[idx] if self.ids is not None else ""
        }

    def __len__(self):
        return self.demo.shape[0]

import os
import torch
from tqdm import tqdm

@torch.no_grad()
def encode_and_save(model_clip, tokenizer, text_batch, date_batch,
                    device, max_events, pca_dim, prefix, batch_idx, save_dir):
    """
    编码一个batch，保存结果为pt文件
    """
    # 解析文本+日期
    tokens_list = []
    dates_list = []
    token_lens = []
    for txt, dt in zip(text_batch, date_batch):
        if txt is None or str(txt) in ['nan', 'None', '']:
            tokens = []
        else:
            tokens = str(txt).split('|')
        if dt is None or str(dt) in ['nan', 'None', '']:
            times = ['0'] * len(tokens)
        else:
            times = str(dt).split('|')
            if len(times) < len(tokens):
                times += ['0'] * (len(tokens) - len(times))
        n = min(len(tokens), len(times))
        tokens = tokens[:n]
        times = times[:n]
        tokens_list.extend(tokens)
        dates_list.append(times)
        token_lens.append(n)

    if len(tokens_list) == 0:
        # 全空情况直接保存0张量
        zero_tensor = torch.zeros((len(text_batch), max_events, 2 + pca_dim), dtype=torch.float32)
        torch.save(zero_tensor, os.path.join(save_dir, f"{prefix}_batch{batch_idx}.pt"))
        return

    # 分批次编码，batch_size尽量小
    batch_size_encode = 64  # 你可以调小试试
    features = []
    for i in range(0, len(tokens_list), batch_size_encode):
        batch_tokens = tokens_list[i:i+batch_size_encode]
        tokens_tensor = tokenizer(batch_tokens).to(device)
        with torch.no_grad():
            emb = model_clip.encode_text(tokens_tensor).float()
        features.append(emb.cpu())
        del tokens_tensor, emb
        torch.cuda.empty_cache()

    features = torch.cat(features, dim=0)

    # PCA降维
    def torch_pca_batch(x, n_components):
        x = x - x.mean(dim=0, keepdim=True)
        U, S, Vt = torch.linalg.svd(x, full_matrices=False)
        return x @ Vt[:n_components].T
    features_pca = torch_pca_batch(features.to(device), pca_dim).cpu()

    # 按参与者组装
    results = []
    idx = 0
    for n, times in zip(token_lens, dates_list):
        arr = torch.zeros((max_events, 2 + pca_dim), dtype=torch.float32)
        if n > 0:
            vecs = features_pca[idx:idx+n]
            idx += n
            for j in range(min(n, max_events)):
                arr[j, 0] = j
                try:
                    arr[j, 1] = float(times[j])
                except:
                    arr[j, 1] = 0
                arr[j, 2:] = vecs[j]
        results.append(arr)
    result_tensor = torch.stack(results, dim=0)
    torch.save(result_tensor, os.path.join(save_dir, f"{prefix}_batch{batch_idx}.pt"))
    del features, features_pca, result_tensor
    torch.cuda.empty_cache()

import json
def vectorize_all_series_resume(
    model_clip, tokenizer,
    phecode_text, phecode_date,
    phenotype_text, phenotype_date,
    drug_text, drug_date,
    opcs4_text, opcs4_date,
    device='cuda', max_events=5, embed_dim=512, pca_dim=62,
    batch_size=512, save_dir="temp_batches"
):
    """
    四类数据同步批处理，断点续跑
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_file = os.path.join(save_dir, "checkpoint.json")

    # 加载断点信息
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
        completed_batches = set(checkpoint.get("completed_batches", []))
    else:
        completed_batches = set()

    total_n = len(phecode_text)
    assert total_n == len(phenotype_text) == len(drug_text) == len(opcs4_text), "输入长度不一致！"

    total_batches = (total_n + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(total_batches),desc='Processing batches'):
        print(f"Processing batch {batch_idx + 1}/{total_batches}")
        if batch_idx in completed_batches:
            continue  # 跳过已完成的批次

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_n)

        # 同时取出四个数据类型的当前批次
        phecode_batch_t = phecode_text[start_idx:end_idx]
        phecode_batch_d = phecode_date[start_idx:end_idx]

        phenotype_batch_t = phenotype_text[start_idx:end_idx]
        phenotype_batch_d = phenotype_date[start_idx:end_idx]

        drug_batch_t = drug_text[start_idx:end_idx]
        drug_batch_d = drug_date[start_idx:end_idx]

        opcs4_batch_t = opcs4_text[start_idx:end_idx]
        opcs4_batch_d = opcs4_date[start_idx:end_idx]

        # 分别处理并保存
        encode_and_save(model_clip, tokenizer, phecode_batch_t, phecode_batch_d,
                        device, max_events, pca_dim, "phecode", batch_idx, save_dir)

        encode_and_save(model_clip, tokenizer, phenotype_batch_t, phenotype_batch_d,
                        device, max_events, pca_dim, "phenotypes", batch_idx, save_dir)

        encode_and_save(model_clip, tokenizer, drug_batch_t, drug_batch_d,
                        device, max_events, pca_dim, "drug", batch_idx, save_dir)

        encode_and_save(model_clip, tokenizer, opcs4_batch_t, opcs4_batch_d,
                        device, max_events, pca_dim, "OPCS4", batch_idx, save_dir)

        # 更新断点
        completed_batches.add(batch_idx)
        with open(checkpoint_file, "w") as f:
            json.dump({"completed_batches": sorted(list(completed_batches))}, f)

        print(f"Batch {batch_idx+1}/{total_batches} completed and saved.")

    print("✅ All batches completed.")

import glob
def merge_pt_batches(save_dir, prefix, output_path):
    """
    合并多个批次的.pt文件，保存为一个大tensor
    prefix: 对应文件名前缀，例如 'phecode', 'phenotypes', 'drug', 'OPCS4'
    output_path: 合并后保存路径
    """
    batch_files = sorted(
        glob.glob(os.path.join(save_dir, f"{prefix}_batch*.pt")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('batch')[-1])
    )

    tensors = []
    for f in tqdm(batch_files, desc='merge_pt_batches'):
        tensors.append(torch.load(f))
    full_tensor = torch.cat(tensors, dim=0)

    torch.save(full_tensor, output_path)
    print(f"✅ Merged {len(batch_files)} batches for '{prefix}' → {output_path} (shape: {full_tensor.shape})")

    return full_tensor


group_map = {
        # ----------- Demographics -----------
        # 'Participant ID': 'Demographics',
        'Sex': 'Demographics',
        'Age at recruitment': 'Demographics',
        'Ethnic background': 'Demographics',
        'Qualifications | Instance 0': 'Demographics',
        'Current employment status | Instance 0': 'Demographics',
        'Age completed full time education | Instance 0': 'Demographics',
        'Genetic kinship to other participants': 'Demographics',
        'Part of a multiple birth | Instance 0': 'Demographics',

        # ----------- Physical Measures -----------
        'Standing height | Instance 0': 'Physical Measures',
        'Seated height | Instance 0': 'Physical Measures',
        'Seating box height | Instance 0': 'Physical Measures',
        'Waist circumference | Instance 0': 'Physical Measures',
        'Hip circumference | Instance 0': 'Physical Measures',
        'Waist-hip circumference ratio | Instance 0': 'Physical Measures',
        'Ankle spacing width | Instance 0': 'Physical Measures',
        'Heel bone mineral density (BMD) | Instance 0': 'Physical Measures',
        'Hand grip strength (average) | Instance 0': 'Physical Measures',
        'Hand grip strength (left) | Instance 0': 'Physical Measures',
        'Hand grip strength (right) | Instance 0': 'Physical Measures',
        'Body mass index (BMI) | Instance 0': 'Physical Measures',
        'Body mass index (BMI) | Instance 0(participant - p21001_i0)': 'Physical Measures',
        'Body mass index (BMI) | Instance 0(participant - p23104_i0)': 'Physical Measures',
        'Weight | Instance 0': 'Physical Measures',
        'Weight | Instance 0(participant - p21002_i0)': 'Physical Measures',
        'Trunk fat percentage | Instance 0': 'Physical Measures',
        'Trunk fat mass | Instance 0': 'Physical Measures',
        'Trunk fat-free mass | Instance 0': 'Physical Measures',
        'Trunk predicted mass | Instance 0': 'Physical Measures',
        'Whole body fat mass | Instance 0': 'Physical Measures',
        'Whole body fat-free mass | Instance 0': 'Physical Measures',
        'Whole body water mass | Instance 0': 'Physical Measures',
        'Impedance of whole body | Instance 0': 'Physical Measures',

        # ----------- Biomarkers -----------
        # 血液生化、尿液等
        'White blood cell (leukocyte) count | Instance 0': 'Biomarkers',
        'Red blood cell (erythrocyte) count | Instance 0': 'Biomarkers',
        'Haemoglobin concentration | Instance 0': 'Biomarkers',
        'Haematocrit percentage | Instance 0': 'Biomarkers',
        'Mean corpuscular volume | Instance 0': 'Biomarkers',
        'Mean corpuscular haemoglobin | Instance 0': 'Biomarkers',
        'Mean corpuscular haemoglobin concentration | Instance 0': 'Biomarkers',
        'Red blood cell (erythrocyte) distribution width | Instance 0': 'Biomarkers',
        'Platelet count | Instance 0': 'Biomarkers',
        'Platelet crit | Instance 0': 'Biomarkers',
        'Mean platelet (thrombocyte) volume | Instance 0': 'Biomarkers',
        'Platelet distribution width | Instance 0': 'Biomarkers',
        'Lymphocyte count | Instance 0': 'Biomarkers',
        'Monocyte count | Instance 0': 'Biomarkers',
        'Neutrophill count | Instance 0': 'Biomarkers',
        'Eosinophill count | Instance 0': 'Biomarkers',
        'Basophill count | Instance 0': 'Biomarkers',
        'Nucleated red blood cell count | Instance 0': 'Biomarkers',
        'Lymphocyte percentage | Instance 0': 'Biomarkers',
        'Monocyte percentage | Instance 0': 'Biomarkers',
        'Neutrophill percentage | Instance 0': 'Biomarkers',
        'Eosinophill percentage | Instance 0': 'Biomarkers',
        'Basophill percentage | Instance 0': 'Biomarkers',
        'Nucleated red blood cell percentage | Instance 0': 'Biomarkers',
        'Reticulocyte percentage | Instance 0': 'Biomarkers',
        'Reticulocyte count | Instance 0': 'Biomarkers',
        'Mean reticulocyte volume | Instance 0': 'Biomarkers',
        'Mean sphered cell volume | Instance 0': 'Biomarkers',
        'Immature reticulocyte fraction | Instance 0': 'Biomarkers',
        'High light scatter reticulocyte percentage | Instance 0': 'Biomarkers',
        'High light scatter reticulocyte count | Instance 0': 'Biomarkers',
        'Creatinine (enzymatic) in urine | Instance 0': 'Biomarkers',
        'Potassium in urine | Instance 0': 'Biomarkers',
        'Sodium in urine | Instance 0': 'Biomarkers',
        'Albumin | Instance 0(participant - p30600_i0)': 'Biomarkers',
        'Alkaline phosphatase | Instance 0': 'Biomarkers',
        'Alanine aminotransferase | Instance 0': 'Biomarkers',
        'Apolipoprotein A | Instance 0': 'Biomarkers',
        'Apolipoprotein B | Instance 0': 'Biomarkers',
        'Aspartate aminotransferase | Instance 0': 'Biomarkers',
        'Direct bilirubin | Instance 0': 'Biomarkers',
        'Urea | Instance 0': 'Biomarkers',
        'Calcium | Instance 0': 'Biomarkers',
        'Cholesterol | Instance 0': 'Biomarkers',
        'Creatinine | Instance 0': 'Biomarkers',
        'C-reactive protein | Instance 0': 'Biomarkers',
        'Cystatin C | Instance 0': 'Biomarkers',
        'Gamma glutamyltransferase | Instance 0': 'Biomarkers',
        'Glucose | Instance 0': 'Biomarkers',
        'Glycated haemoglobin (HbA1c) | Instance 0': 'Biomarkers',
        'HDL cholesterol | Instance 0': 'Biomarkers',
        'IGF-1 | Instance 0': 'Biomarkers',
        'LDL direct | Instance 0': 'Biomarkers',
        'Lipoprotein A | Instance 0': 'Biomarkers',
        'Phosphate | Instance 0': 'Biomarkers',
        'SHBG | Instance 0': 'Biomarkers',
        'Total bilirubin | Instance 0': 'Biomarkers',
        'Testosterone | Instance 0': 'Biomarkers',
        'Total protein | Instance 0': 'Biomarkers',
        'Triglycerides | Instance 0': 'Biomarkers',
        'Urate | Instance 0': 'Biomarkers',
        'Vitamin D | Instance 0': 'Biomarkers',

        # ----------- Lifestyle Factors -----------
        'Alcohol intake frequency. | Instance 0': 'Lifestyle Factors',
        'Alcohol drinker status | Instance 0': 'Lifestyle Factors',
        'Alcohol usually taken with meals | Instance 0': 'Lifestyle Factors',
        'Alcohol intake versus 10 years previously | Instance 0': 'Lifestyle Factors',
        'Current tobacco smoking | Instance 0': 'Lifestyle Factors',
        'Smoking status | Instance 0': 'Lifestyle Factors',
        'Ever smoked | Instance 0': 'Lifestyle Factors',
        'Past tobacco smoking | Instance 0': 'Lifestyle Factors',
        'Exposure to tobacco smoke at home | Instance 0': 'Lifestyle Factors',
        'Exposure to tobacco smoke outside home | Instance 0': 'Lifestyle Factors',
        'Weekly usage of mobile phone in last 3 months | Instance 0': 'Lifestyle Factors',
        'Hands-free device/speakerphone use with mobile phone in last 3 month | Instance 0': 'Lifestyle Factors',
        'Difference in mobile phone use compared to two years previously | Instance 0': 'Lifestyle Factors',
        'Usual side of head for mobile phone use | Instance 0': 'Lifestyle Factors',
        'Plays computer games | Instance 0': 'Lifestyle Factors',
        'Time spent watching television (TV) | Instance 0': 'Lifestyle Factors',
        'Time spent using computer | Instance 0': 'Lifestyle Factors',
        'Time spent driving | Instance 0': 'Lifestyle Factors',
        'Nap during day | Instance 0': 'Lifestyle Factors',
        'Snoring | Instance 0': 'Lifestyle Factors',
        'Daytime dozing / sleeping | Instance 0': 'Lifestyle Factors',
        'Duration of walks | Instance 0': 'Lifestyle Factors',
        'Duration of moderate activity | Instance 0': 'Lifestyle Factors',
        'Frequency of stair climbing in last 4 weeks | Instance 0': 'Lifestyle Factors',
        'Frequency of walking for pleasure in last 4 weeks | Instance 0': 'Lifestyle Factors',
        'Duration walking for pleasure | Instance 0': 'Lifestyle Factors',
        'Number of days/week walked 10+ minutes | Instance 0': 'Lifestyle Factors',
        'Number of days/week of moderate physical activity 10+ minutes | Instance 0': 'Lifestyle Factors',
        'Number of days/week of vigorous physical activity 10+ minutes | Instance 0': 'Lifestyle Factors',
        'Types of transport used (excluding work) | Instance 0': 'Lifestyle Factors',
        'Types of physical activity in last 4 weeks | Instance 0': 'Lifestyle Factors',
        'Leisure/social activities | Instance 0': 'Lifestyle Factors',
        'Sleep duration | Instance 0': 'Lifestyle Factors',
        'Sleeplessness / insomnia | Instance 0': 'Lifestyle Factors',

        # 饮食相关
        'Cooked vegetable intake | Instance 0': 'Lifestyle Factors',
        'Salad / raw vegetable intake | Instance 0': 'Lifestyle Factors',
        'Fresh fruit intake | Instance 0': 'Lifestyle Factors',
        'Dried fruit intake | Instance 0': 'Lifestyle Factors',
        'Oily fish intake | Instance 0': 'Lifestyle Factors',
        'Non-oily fish intake | Instance 0': 'Lifestyle Factors',
        'Processed meat intake | Instance 0': 'Lifestyle Factors',
        'Poultry intake | Instance 0': 'Lifestyle Factors',
        'Beef intake | Instance 0': 'Lifestyle Factors',
        'Lamb/mutton intake | Instance 0': 'Lifestyle Factors',
        'Pork intake | Instance 0': 'Lifestyle Factors',
        'Cheese intake | Instance 0': 'Lifestyle Factors',
        'Milk type used | Instance 0': 'Lifestyle Factors',
        'Spread type | Instance 0': 'Lifestyle Factors',
        'Bread intake | Instance 0': 'Lifestyle Factors',
        'Bread type | Instance 0': 'Lifestyle Factors',
        'Cereal intake | Instance 0': 'Lifestyle Factors',
        'Cereal type | Instance 0': 'Lifestyle Factors',
        'Tea intake | Instance 0': 'Lifestyle Factors',
        'Coffee type | Instance 0': 'Lifestyle Factors',
        'Hot drink temperature | Instance 0': 'Lifestyle Factors',
        'Water intake | Instance 0': 'Lifestyle Factors',
        'Salt added to food | Instance 0': 'Lifestyle Factors',
        'Never eat eggs, dairy, wheat, sugar | Instance 0': 'Lifestyle Factors',

        # ----------- Environmental Exposures -----------
        'Greenspace percentage, buffer 1000m | Instance 0': 'Environmental Exposures',
        'Type of accommodation lived in | Instance 0': 'Environmental Exposures',
        'Own or rent accommodation lived in | Instance 0': 'Environmental Exposures',
        'Number in household | Instance 0': 'Environmental Exposures',
        'Number of vehicles in household | Instance 0': 'Environmental Exposures',
        'Length of time at current address | Instance 0': 'Environmental Exposures',
        'Gas or solid-fuel cooking/heating | Instance 0': 'Environmental Exposures',
        'Attendance/disability/mobility allowance | Instance 0': 'Environmental Exposures',

        # ----------- Mental Health and Psychosocial Factors -----------
        'Mood swings | Instance 0': 'Mental Health and Psychosocial Factors',
        'Miserableness | Instance 0': 'Mental Health and Psychosocial Factors',
        'Nervous feelings | Instance 0': 'Mental Health and Psychosocial Factors',
        'Worrier / anxious feelings | Instance 0': 'Mental Health and Psychosocial Factors',
        'Tense / \'highly strung\' | Instance 0': 'Mental Health and Psychosocial Factors',
        'Frequency of depressed mood in last 2 weeks | Instance 0': 'Mental Health and Psychosocial Factors',
        'Frequency of unenthusiasm / disinterest in last 2 weeks | Instance 0': 'Mental Health and Psychosocial Factors',
        'Frequency of tenseness / restlessness in last 2 weeks | Instance 0': 'Mental Health and Psychosocial Factors',
        'Frequency of tiredness / lethargy in last 2 weeks | Instance 0': 'Mental Health and Psychosocial Factors',
        'Guilty feelings | Instance 0': 'Mental Health and Psychosocial Factors',
        'Loneliness, isolation | Instance 0': 'Mental Health and Psychosocial Factors',
        'Irritability | Instance 0': 'Mental Health and Psychosocial Factors',
        'Sensitivity / hurt feelings | Instance 0': 'Mental Health and Psychosocial Factors',
        'Fed-up feelings | Instance 0': 'Mental Health and Psychosocial Factors',
        'Worry too long after embarrassment | Instance 0': 'Mental Health and Psychosocial Factors',
        'Suffer from \'nerves\' | Instance 0': 'Mental Health and Psychosocial Factors',
        'Risk taking | Instance 0': 'Mental Health and Psychosocial Factors',
        'Seen doctor (GP) for nerves, anxiety, tension or depression | Instance 0': 'Mental Health and Psychosocial Factors',
        'Seen a psychiatrist for nerves, anxiety, tension or depression | Instance 0': 'Mental Health and Psychosocial Factors',
        'Illness, injury, bereavement, stress in last 2 years | Instance 0': 'Mental Health and Psychosocial Factors',
        'Able to confide | Instance 0': 'Mental Health and Psychosocial Factors',
        'Frequency of friend/family visits | Instance 0': 'Mental Health and Psychosocial Factors',

        # ----------- Diagnosis (Medical Records) -----------
        # EHR: ICD, phecode, baseline日期等
        'Baseline_Date_x': 'Diagnosis (Medical Records)',
        'Before_Baseline-ICD': 'Diagnosis (Medical Records)',
        'Before_Baseline-Disease': 'Diagnosis (Medical Records)',
        'Before_Baseline-ICD_Date': 'Diagnosis (Medical Records)',
        'Before_Baseline-ICD_Days_To_Baseline': 'Diagnosis (Medical Records)',
        'Before_Baseline-PheCode': 'Diagnosis (Medical Records)',
        'Before_Baseline-Phenotype': 'Diagnosis (Medical Records)',
        'Before_Baseline-Category': 'Diagnosis (Medical Records)',
        'After_Baseline-ICD': 'Diagnosis (Medical Records)',
        'After_Baseline-Disease': 'Diagnosis (Medical Records)',
        'After_Baseline-ICD_Date': 'Diagnosis (Medical Records)',
        'After_Baseline-ICD_Days_To_Baseline': 'Diagnosis (Medical Records)',
        'After_Baseline-PheCode': 'Diagnosis (Medical Records)',
        'After_Baseline-Phenotype': 'Diagnosis (Medical Records)',
        'After_Baseline-Category': 'Diagnosis (Medical Records)',
        'Phecode_Output_Within5Years': 'Diagnosis (Medical Records)',
        'Phecode_Output_Within5Years_Days': 'Diagnosis (Medical Records)',
        'y_auc':'Diagnosis (Medical Records)',
        'Y_time':'Diagnosis (Medical Records)',
        'Y_event':'Diagnosis (Medical Records)',

        # ----------- Medications (Medical Records) -----------
        'Baseline_Date_y': 'Medications (Medical Records)',
        'Before_Baseline-Drug': 'Medications (Medical Records)',
        'Before_Baseline-Drug_Date': 'Medications (Medical Records)',
        'Before_Baseline-Drug_Quantity': 'Medications (Medical Records)',
        'Before_Baseline-Drug_Days_To_Baseline': 'Medications (Medical Records)',
        'After_Baseline-Drug': 'Medications (Medical Records)',
        'After_Baseline-Drug_Date': 'Medications (Medical Records)',
        'After_Baseline-Drug_Quantity': 'Medications (Medical Records)',
        'After_Baseline-Drug_Days_To_Baseline': 'Medications (Medical Records)',

        # ----------- Operative Procedures (Medical Records) -----------
        'Baseline_Date': 'Operative Procedures (Medical Records)',
        'Before_Baseline-OPCS4': 'Operative Procedures (Medical Records)',
        'Before_Baseline-OPCS4_Date': 'Operative Procedures (Medical Records)',
        'Before_Baseline-OPCS4_Days_To_Baseline': 'Operative Procedures (Medical Records)',
        'After_Baseline-OPCS4': 'Operative Procedures (Medical Records)',
        'After_Baseline-OPCS4_Date': 'Operative Procedures (Medical Records)',
        'After_Baseline-OPCS4_Days_To_Baseline': 'Operative Procedures (Medical Records)',

    }
