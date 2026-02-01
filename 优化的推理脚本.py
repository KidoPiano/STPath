import os
import glob
import torch
import multiprocessing as mp
from stpath.hest_utils.file_utils import read_assets_from_h5
from stpath.app.pipeline.inference import STPathInference
import argparse

# ==========================================
# 【用户配置区】 直接在这里修改你的默认路径
# ==========================================
DEFAULT_INPUT = "/mnt/net_sda/rst/Data_for_Survival_yzy/20x_256px_0px_overlap/features_gigapath"
DEFAULT_OUTPUT = "/mnt/net_sda/rst/STPath/MBC_survival_yzy/MBC_gigapath_STPath_pred"
DEFAULT_MODE = "single"  # "root" 遍历子目录, "single" 处理当前目录
DEFAULT_GPUS = "all"
""   # "all" 或指定 "0,1,2"

# 模型相关路径
GENE_VOC_PATH = '/mnt/sdb/yzy/STPath/utils_data/symbol2ensembl.json'
MODEL_WEIGHT_PATH = '/home/yzy/.cache/huggingface/hub/models--tlhuang--STPath/snapshots/3346881771f2ddb5575532df3df1b5477846d10a/stfm.pth'

# 显存碎片优化
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

# --- 核心工作进程 ---
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
        if task is None:
            break
        
        ds_name, emb_file_path = task
        sample_id = os.path.splitext(os.path.basename(emb_file_path))[0].replace("_features", "")
        
        current_pred_dir = os.path.join(prediction_root, ds_name)
        os.makedirs(current_pred_dir, exist_ok=True)
        save_path = os.path.join(current_pred_dir, f"pred_{sample_id}.h5ad")

        if os.path.exists(save_path):
            print(f"[GPU {gpu_id}] 跳过已存在: {sample_id}")
            continue

        try:
            config = get_config(ds_name)
            data_dict, _ = read_assets_from_h5(emb_file_path)
            
            # 执行推理
            pred_adata = agent.inference(
                coords=data_dict["coords"], 
                img_features=data_dict["features"], 
                organ_type=config["organ_type"], 
                tech_type=config["tech_type"],
                save_gene_names=None
            )
            pred_adata.write_h5ad(save_path)
            
            # 及时释放显存
            del pred_adata
            torch.cuda.empty_cache()
            print(f"[GPU {gpu_id}] SUCCESS: {sample_id}")
            
        except Exception as e:
            print(f"[GPU {gpu_id} ERROR] 样本 {sample_id} 失败: {e}")
    
    print(f"[Worker] GPU {gpu_id} 任务结束。")

# --- 主逻辑 ---
def main():
    parser = argparse.ArgumentParser(description="STPath Multi-GPU Inference Script")
    # 去掉了 required=True，并将默认值指向了【用户配置区】的变量
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", type=str, choices=['root', 'single'], default=DEFAULT_MODE)
    parser.add_argument("--gpus", type=str, default=DEFAULT_GPUS)
    
    args = parser.parse_args()

    # 1. GPU 列表设置
    if args.gpus == "all":
        num_gpus = torch.cuda.device_count()
        gpu_list = list(range(num_gpus))
    else:
        gpu_list = [int(x) for x in args.gpus.split(",")]

    # 2. 任务收集
    all_tasks = []
    if args.mode == 'root':
        if not os.path.exists(args.input):
            print(f"[ERROR] 输入路径不存在: {args.input}")
            return
        ds_names = [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
        for ds in ds_names:
            h5_files = glob.glob(os.path.join(args.input, ds, "*.h5"))
            for f in h5_files:
                all_tasks.append((ds, f))
    else:
        ds_name = os.path.basename(args.input.rstrip('/'))
        h5_files = glob.glob(os.path.join(args.input, "*.h5"))
        for f in h5_files:
            all_tasks.append((ds_name, f))

    if not all_tasks:
        print(f"[Exit] 未找到任何 .h5 文件。当前搜索路径: {args.input}")
        return

    print(f"--- 任务启动 ---")
    print(f"输入路径: {args.input}")
    print(f"输出路径: {args.output}")
    print(f"使用 GPU: {gpu_list}")
    print(f"总任务数: {len(all_tasks)}")
    print(f"----------------")

    # 3. 队列分发
    task_queue = mp.Queue()
    for task in all_tasks:
        task_queue.put(task)
    for _ in range(len(gpu_list)):
        task_queue.put(None)

    # 4. 启动多进程
    processes = []
    for gid in gpu_list:
        p = mp.Process(target=worker, args=(gid, task_queue, args.output))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n[FINISH] 所有任务已完成！")

if __name__ == "__main__":
    # 多进程必须使用 spawn 模式
    mp.set_start_method('spawn', force=True) 
    main()