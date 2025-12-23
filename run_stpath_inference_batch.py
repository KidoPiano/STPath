# 此脚本用于批量运行 STPath 推理任务，适用于聚类任务的多个数据集。
# 从h5文件中读取嵌入特征，送入STPath模型进行推理，得到h5ad文件。
# 此脚本依赖STPath库，请确保已正确安装并配置环境，需放在STPath项目目录下运行。

import os
import glob
from stpath.hest_utils.file_utils import read_assets_from_h5
from stpath.app.pipeline.inference import STPathInference

# --- 核心配置：模型和路径 ---
gene_voc_path = '/mnt/sdb/yzy/STPath/utils_data/symbol2ensembl.json'
model_weight_path = '/home/yzy/.cache/huggingface/hub/models--tlhuang--STPath/snapshots/3346881771f2ddb5575532df3df1b5477846d10a/stfm.pth'
device = 0  # GPU 设备ID

# --- 批量推理函数 ---
def run_stpath_multi_dataset_inference(
    embedding_root: str, 
    prediction_root: str
):
    """
    遍历 embedding_root 下的所有子文件夹（数据集），运行 STPath 推理。
    """
    # 1. 初始化 STPath Agent (全局只需一次)
    print("="*60)
    print(f"[INFO] 正在初始化 STPath 推理 Agent (Device: {device})...")
    try:
        agent = STPathInference(
            gene_voc_path=gene_voc_path,
            model_weight_path=model_weight_path,
            device=device)
    except Exception as e:
        print(f"[ERROR] 无法初始化 STPath Agent: {e}")
        return

    # 2. 定义数据集特有的参数映射 (如果不在此字典中，则使用默认值)
    # 这里的 tech_type 必须符合 STPath 要求：["<pad>", "Spatial Transcriptomics", "Visium", "Xenium", "Visium HD"]
    dataset_configs = {
        "Andersson_ST": {"tech_type": "Spatial Transcriptomics", "organ_type": "Breast"},
        "Maynard_visium": {"tech_type": "Visium", "organ_type": "Brain"}, # Maynard 通常是脑皮层
        "GSE213688_visium": {"tech_type": "Visium", "organ_type": "Breast"},
        "Erickson_visium": {"tech_type": "Visium", "organ_type": "Prostate"},
        "Chen_": {"tech_type": "Visium", "organ_type": "Breast"}
    }

    # 3. 开始遍历数据集文件夹
    # 获取 embedding_root 下的所有子目录名
    ds_names = [d for d in os.listdir(embedding_root) if os.path.isdir(os.path.join(embedding_root, d))]
    
    for ds_name in ds_names:
        print(f"\n\n{'#'*30}\n[DATASET] 正在处理数据集: {ds_name}\n{'#'*30}")
        
        # 获取当前数据集配置
        config = dataset_configs.get(ds_name, {"tech_type": "Visium", "organ_type": "Breast"})
        cur_tech = config["tech_type"]
        cur_organ = config["organ_type"]

        # 构建输入和输出路径
        current_emb_dir = os.path.join(embedding_root, ds_name)
        current_pred_dir = os.path.join(prediction_root, ds_name)
        os.makedirs(current_pred_dir, exist_ok=True)

        # 查找该数据集下的所有 .h5 文件
        h5_files = glob.glob(os.path.join(current_emb_dir, "*.h5"))
        print(f"[INFO] 找到 {len(h5_files)} 个样本。技术类型: {cur_tech}, 组织类型: {cur_organ}")

        if not h5_files:
            print(f"[SKIP] {ds_name} 目录下没有 .h5 文件，跳过。")
            continue

        # 4. 循环处理数据集内的每个样本
        for i, emb_file_path in enumerate(h5_files):
            sample_id = os.path.splitext(os.path.basename(emb_file_path))[0].replace("_features", "")
            save_path = os.path.join(current_pred_dir, f"pred_{sample_id}.h5ad")

            # 如果已经存在，可以选择跳过（断点续跑）
            if os.path.exists(save_path):
                print(f"  [{i+1}/{len(h5_files)}] 样本 {sample_id} 已存在预测结果，跳过。")
                continue

            print(f"  [{i+1}/{len(h5_files)}] 正在处理样本: {sample_id}")
            
            try:
                # 4.1 加载特征
                data_dict, _ = read_assets_from_h5(emb_file_path)
                coords = data_dict["coords"]
                embeddings = data_dict["features"]

                # 4.2 推理
                pred_adata = agent.inference(
                    coords=coords, 
                    img_features=embeddings, 
                    organ_type=cur_organ, 
                    tech_type=cur_tech,
                    save_gene_names=None
                )

                # 4.3 保存
                pred_adata.write_h5ad(save_path)
                print(f"    [SUCCESS] 已保存到: {save_path}")

            except Exception as e:
                print(f"    [ERROR] 样本 {sample_id} 处理失败: {e}")
                continue

    print("\n[FINISH] 所有数据集推理任务完成！")

# --- 运行块 ---
if __name__ == "__main__":
    # 根据你的描述设定的根目录
    EMB_ROOT = "/mnt/net_sda/rst/Sub_dataset_for_spatial_cluster_yzy/embedding_results"
    PRED_ROOT = "/mnt/net_sda/rst/Sub_dataset_for_spatial_cluster_yzy/prediction_results_STPath"

    run_stpath_multi_dataset_inference(EMB_ROOT, PRED_ROOT)