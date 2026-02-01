import os
import glob
import torch
import numpy as np
import multiprocessing as mp
import anndata as ad  # 必须安装 anndata
from stpath.hest_utils.file_utils import read_assets_from_h5
from stpath.app.pipeline.inference import STPathInference
import argparse

# ==========================================
# 【用户配置区】
# ==========================================
DEFAULT_INPUT = "/mnt/net_sda/rst/Data_for_Survival_yzy/20x_256px_0px_overlap/features_gigapath"
DEFAULT_OUTPUT = "/mnt/net_sda/rst/STPath/MBC_survival_yzy/MBC_gigapath_STPath_pred"
DEFAULT_MODE = "single"
DEFAULT_GPUS = "0" 

GENE_VOC_PATH = '/mnt/sdb/yzy/STPath/utils_data/symbol2ensembl.json'
MODEL_WEIGHT_PATH = '/home/yzy/.cache/huggingface/hub/models--tlhuang--STPath/snapshots/3346881771f2ddb5575532df3df1b5477846d10a/stfm.pth'

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DATASET_CONFIGS = {
    "Andersson_ST": {"tech_type": "Spatial Transcriptomics", "organ_type": "Breast"},
    "Maynard_visium": {"tech_type": "Visium", "organ_type": "Brain"},
    "GSE213688_visium": {"tech_type": "Visium", "organ_type": "Breast"},
    "Erickson_visium": {"tech_type": "Visium", "organ_type": "Prostate"},
    "Chen_": {"tech_type": "Visium", "organ_type": "Breast"}
}

def get_config(ds_name):
    return DATASET_CONFIGS.get(ds_name, {"tech_type": "Visium", "organ_type": "Breast"})

def worker(gpu_id, task_queue, prediction_root):
    print(f"[Worker] GPU {gpu_id} 正在初始化...")
    try:
        agent = STPathInference(
            gene_voc_path=GENE_VOC_PATH,
            model_weight_path=MODEL_WEIGHT_PATH,
            device=gpu_id
        )
    except Exception as e:
        print(f"[Worker ERROR] GPU {gpu_id} 初始化失败: {e}")
        return

    while True:
        task = task_queue.get()
        if task is None: break
        
        ds_name, emb_file_path = task
        sample_id = os.path.splitext(os.path.basename(emb_file_path))[0].replace("_features", "")
        current_pred_dir = os.path.join(prediction_root, ds_name)
        os.makedirs(current_pred_dir, exist_ok=True)
        save_path = os.path.join(current_pred_dir, f"pred_{sample_id}.h5ad")

        if os.path.exists(save_path):
            print(f"[GPU {gpu_id}] 跳过: {sample_id}")
            continue

        try:
            config = get_config(ds_name)
            data_dict, _ = read_assets_from_h5(emb_file_path)
            coords = data_dict["coords"]
            features = data_dict["features"]
            num_patches = len(features)

            # --- 分块推理逻辑 (核心修改) ---
            MAX_CHUNK = 8000 # 设为 8000 保证即使显存有碎片也能跑通
            if num_patches > MAX_CHUNK:
                print(f"[GPU {gpu_id}] 样本 {sample_id} 过大({num_patches} patches)，启动分块推理...")
                chunk_adatas = []
                for start_idx in range(0, num_patches, MAX_CHUNK):
                    end_idx = min(start_idx + MAX_CHUNK, num_patches)
                    c_slice = coords[start_idx:end_idx]
                    f_slice = features[start_idx:end_idx]
                    
                    with torch.no_grad():
                        # 每一块分别推理
                        chunk_out = agent.inference(
                            coords=c_slice, 
                            img_features=f_slice, 
                            organ_type=config["organ_type"], 
                            tech_type=config["tech_type"],
                            save_gene_names=None
                        )
                        chunk_adatas.append(chunk_out)
                
                # 拼接所有块，确保 N 依然是原来的总数
                pred_adata = ad.concat(chunk_adatas, axis=0)
            else:
                # 正常推理
                with torch.no_grad():
                    pred_adata = agent.inference(
                        coords=coords, 
                        img_features=features, 
                        organ_type=config["organ_type"], 
                        tech_type=config["tech_type"],
                        save_gene_names=None
                    )
            
            pred_adata.write_h5ad(save_path)
            del pred_adata
            torch.cuda.empty_cache()
            print(f"[GPU {gpu_id}] SUCCESS: {sample_id} ({num_patches} patches)")
            
        except Exception as e:
            print(f"[GPU {gpu_id} ERROR] 样本 {sample_id} 失败: {e}")
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", type=str, choices=['root', 'single'], default=DEFAULT_MODE)
    parser.add_argument("--gpus", type=str, default=DEFAULT_GPUS)
    args = parser.parse_args()

    gpu_list = [int(x) for x in args.gpus.split(",")] if args.gpus != "all" else list(range(torch.cuda.device_count()))

    all_tasks = []
    if args.mode == 'root':
        ds_names = [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
        for ds in ds_names:
            for f in glob.glob(os.path.join(args.input, ds, "*.h5")):
                all_tasks.append((ds, f))
    else:
        ds_name = os.path.basename(args.input.rstrip('/'))
        for f in glob.glob(os.path.join(args.input, "*.h5")):
            all_tasks.append((ds_name, f))

    task_queue = mp.Queue()
    for t in all_tasks: task_queue.put(t)
    for _ in range(len(gpu_list)): task_queue.put(None)

    processes = []
    for gid in gpu_list:
        p = mp.Process(target=worker, args=(gid, task_queue, args.output))
        p.start()
        processes.append(p)
    for p in processes: p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) 
    main()