import os
import glob
import numpy as np
import scanpy as sc
from stpath.hest_utils.file_utils import read_assets_from_h5
from stpath.app.pipeline.inference import STPathInference

# --- 核心配置：模型和数据路径 ---
# 注意：请确保它们在你的环境中是正确的！
gene_voc_path = '/mnt/sdb/yzy/STPath/utils_data/symbol2ensembl.json'
model_weight_path = '/home/yzy/.cache/huggingface/hub/models--tlhuang--STPath/snapshots/3346881771f2ddb5575532df3df1b5477846d10a/stfm.pth'
device = 0 # GPU 设备ID  没有GPU就填 cpu

# --- 批量推理函数 ---
def run_stpath_inference_batch(
    embedding_path: str, 
    output_path: str, 
    organ_type: str,  
    tech_type: str = "Visium"
):
    """
    批量对指定目录下的所有 .h5 特征文件进行 STPath 推理，并保存为 .h5ad 文件。
    Args:
        embedding_path (str): 包含所有 Gigapath 提取的 .h5 特征文件的目录。
        output_path (str): 推理结果 .h5ad 文件的保存目录。
        organ_type (str): 组织类型 (如 "Breast", "Prostate")，用于模型输入。
        tech_type (str): 测序技术类型 (如 "Visium")，用于模型输入。支持["<pad>", "Spatial Transcriptomics", "Visium", "Xenium", "Visium HD"]
    """
    
    # 1. 初始化 STPathInference Agent (只需初始化一次)
    print("="*60)
    print(f"[INFO] 正在初始化 STPath 推理 Agent (Device: {device})...")
    print(f"  - 模型权重路径: {model_weight_path}")
    print(f"  - 基因词汇表路径: {gene_voc_path}")
    
    try:
        agent = STPathInference(
            gene_voc_path=gene_voc_path,
            model_weight_path=model_weight_path,
            device=device)
    except Exception as e:
        print(f"[ERROR] 无法初始化 STPath Agent，请检查路径和权限: {e}")
        print("="*60)
        return

    # 2. 准备输出目录
    os.makedirs(output_path, exist_ok=True)
    print(f"[INFO] 推理结果将保存到目录: {output_path}")

    # 3. 查找所有 .h5 文件
    h5_files = glob.glob(os.path.join(embedding_path, "*.h5"))           #获取所有.h5文件的路径（列表）
    print(f"[INFO] 在目录 {embedding_path} 中找到 {len(h5_files)} 个 .h5 样本文件。")

    if not h5_files:
        print("[WARNING] 未找到任何 .h5 文件，请检查输入路径是否正确。")
        print("="*60)
        return

    # 4. 循环处理所有文件
    for i, emb_file_path in enumerate(h5_files): 
        file_name = os.path.basename(emb_file_path)      #获取文件名
        sample_id = os.path.splitext(file_name)[0]       #取出不带后缀的文件名作为样本ID
        
        print(f"\n--- [{i+1}/{len(h5_files)}] 正在处理样本: {sample_id} ---")
        
        try:
            # 4.1 加载数据
            print(f"  [STEP] 正在加载特征文件: {file_name}")
            data_dict, _ = read_assets_from_h5(emb_file_path)
            coords = data_dict["coords"]
            embeddings = data_dict["features"]
            
            print(f"  [INFO] 成功加载 {coords.shape[0]} 个 Spots 的特征。")        #coords的形状是spots个行，2列（坐标）

            # 4.2 执行 STPath 推理
            pred_adata = agent.inference(
                coords=coords, 
                img_features=embeddings, 
                organ_type=organ_type, 
                tech_type=tech_type,
                save_gene_names=None  # None 表示保存模型词汇表中的所有 38984 个基因
            )

            # 4.3 保存结果
            save_path = os.path.join(output_path, f"pred_{sample_id}.h5ad")
            pred_adata.write_h5ad(save_path)
            print(f"  [SUCCESS] 推理结果已保存到: {save_path}")

        except Exception as e:
            print(f"  [ERROR] 处理样本 {sample_id} 时发生错误，跳过该文件: {e}")
            continue # 继续处理下一个文件
            
    print("\n[INFO] 所有文件处理完毕。")

# --- 示例运行块 ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch STPath inference")

    parser.add_argument("--embedding_path", type=str,
                        default="/mnt/net_sda/rst/M2OST/HER2+/embedding_yzy/20x_256px_0px_overlap/features_gigapath")    
                                                                    #单独运行此文件时将这里的默认路径改为你实际的路径
    parser.add_argument("--output_path", type=str,
                        default="/mnt/sdb/yzy/MyFiles/STPath/Predict")
                                                                    #单独运行此文件时将这里的默认路径改为你实际的路径

    parser.add_argument("--organ_type", type=str, default="Breast")
    parser.add_argument("--tech_type", type=str, default="Visium")

    args = parser.parse_args()

    run_stpath_inference_batch(
        embedding_path=args.embedding_path,
        output_path=args.output_path,
        organ_type=args.organ_type,
        tech_type=args.tech_type
    )
