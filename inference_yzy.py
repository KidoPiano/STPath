import os
import json
import numpy as np
import scanpy as sc
from scipy.stats import pearsonr
from stpath.hest_utils.st_dataset import load_adata
from stpath.hest_utils.file_utils import read_assets_from_h5
from stpath.app.pipeline.inference import STPathInference


sample_id = "A1"
emb_path = "/mnt/net_sda/rst/M2OST/HER2+/embedding_yzy/20x_256px_0px_overlap/features_gigapath/A1.h5"


#source_dataroot = "/mnt/sdb/yzy/STPath"  # the root directory of the STPath repository
#with open(os.path.join(source_dataroot, "example_data/var_50genes.json")) as f:
#    hvg_list = json.load(f)['genes']

data_dict, _ = read_assets_from_h5(emb_path)  # load the data from the h5 file
coords = data_dict["coords"]
embeddings = data_dict["features"]
#barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
#adata = sc.read_h5ad(os.path.join(source_dataroot, f"{sample_id}.h5ad"))[barcodes, :]



agent = STPathInference(
    gene_voc_path='/mnt/sdb/yzy/STPath/utils_data/symbol2ensembl.json',
    model_weight_path='/home/yzy/.cache/huggingface/hub/models--tlhuang--STPath/snapshots/3346881771f2ddb5575532df3df1b5477846d10a/stfm.pth', 
    device=0)


# The return pred_adata includes the expressions of the genes in hvg_list, which is a list of highly variable genes.
pred_adata = agent.inference(
    coords=coords, 
    img_features=embeddings, 
    organ_type="Breast", 
    tech_type="Visium",
    save_gene_names= None  #  a list of gene names to save in the adata, e.g., ['GATA3', 'UBLE2C', ...]. None will save all genes in the model.
)

# 保存推理结果为 h5ad 文件
save_dir = "/mnt/sdb/yzy/MyFiles/Predict"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"pred_{sample_id}.h5ad")
pred_adata.write_h5ad(save_path)
print(f"[INFO] 推理结果已保存到: {save_path}")

# calculate the Pearson correlation coefficient between the predicted and ground truth gene expression
# all_pearson_list = []
# gt = np.log1p(adata[:, hvg_list].X.toarray())  # sparse -> dense
# go through each gene in the highly variable genes list
# for i in range(len(hvg_list)):
#     pearson_corr, _ = pearsonr(gt[:, i], pred_adata.X[:, i])
#     all_pearson_list.append(pearson_corr.item())
# print(f"Pearson correlation for {sample_id}: {np.mean(all_pearson_list)}")  # 0.1562

