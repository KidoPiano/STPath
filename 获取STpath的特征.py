import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import torch.multiprocessing as mp
from stpath.app.pipeline.inference import STPathInference

# ================= 配置区 =================
CONFIG = {
    "input_dir": "/mnt/net_sda/rst/Data_for_Survival_yzy/20x_256px_0px_overlap/features_gigapath",
    "output_dir": "/mnt/net_sda/rst/STPath/MBC_survival_yzy/MBC_gigapath_STPath_pred",
    "model_weight": "/home/yzy/.cache/huggingface/hub/models--tlhuang--STPath/snapshots/3346881771f2ddb5575532df3df1b5477846d10a/stfm.pth",
    "gene_voc_path": "/mnt/sdb/yzy/STPath/utils_data/symbol2ensembl.json",
    
    "num_processes": 4,        # 建议设为 8 (每卡1进程) 或 16 (每卡2进程)
    "available_gpus": [0,1,2,3], 
    "organ_type": "Breast",      # 鼻咽癌推荐 Mouth
    "tech_type": "Visium"
}

# ================= 工作进程逻辑 =================

def worker(proc_id, task_list, gpu_id, id_to_symbol):
    """
    单个子进程：加载模型 -> 执行推理 -> 保存 CSV
    """
    print(f"[进程 {proc_id}] 正在 GPU:{gpu_id} 上初始化模型...")
    try:
        agent = STPathInference(
            gene_voc_path=CONFIG["gene_voc_path"],
            model_weight_path=CONFIG["model_weight"],
            device=gpu_id
        )
    except Exception as e:
        print(f"[进程 {proc_id}] 初始化失败: {e}")
        return

    for sample_id, pt_path, csv_path in task_list:
        try:
            # 1. 数据准备
            features = torch.load(pt_path, map_location='cpu').numpy().astype(np.float32)
            df_in = pd.read_csv(csv_path)
            # 解析 6272_20608 格式坐标
            coords = np.array([list(map(float, c.split('_'))) for c in df_in['patch_cood']], dtype=np.float32)

            # 2. 模型推理
            adata = agent.inference(
                coords=coords, 
                img_features=features, 
                organ_type=CONFIG["organ_type"], 
                tech_type=CONFIG["tech_type"]
            )

            # 3. 结果持久化
            out_path = Path(CONFIG["output_dir"]) / sample_id
            out_path.mkdir(parents=True, exist_ok=True)
            
            # 保存 H5AD
            adata.write(out_path / f"{sample_id}.h5ad")

            # 保存表达矩阵 CSV (行=基因, 列=Spot)
            exp_data = adata.X.T 
            if hasattr(exp_data, "toarray"): exp_data = exp_data.toarray()
            
            df_exp = pd.DataFrame(exp_data, index=adata.var_names, 
                                  columns=[f"S_{i}" for i in range(adata.n_obs)])
            df_exp.insert(0, 'Gene_Symbol', [id_to_symbol.get(idx, "Unknown") for idx in df_exp.index])
            df_exp.to_csv(out_path / f"{sample_id}_expression.csv")
            
            # 保存坐标 CSV
            df_coords = pd.DataFrame(adata.obsm['coordinates'], columns=['X', 'Y'])
            df_coords.to_csv(out_path / f"{sample_id}_coords.csv", index=False)

            print(f"[进程 {proc_id}] 完成样本: {sample_id}")

        except Exception as e:
            print(f"[进程 {proc_id}] 处理 {sample_id} 发生错误: {e}")

# ================= 主控制程序 =================

def main():
    # 1. 扫描任务
    input_path = Path(CONFIG["input_dir"])
    pt_files = sorted(list(input_path.glob("*.pt")))
    all_tasks = []
    for pf in pt_files:
        sid = pf.stem
        cf = input_path / f"{sid}.csv"
        if cf.exists():
            all_tasks.append((sid, str(pf), str(cf)))

    if not all_tasks:
        print("未找到可处理的任务，请检查路径。")
        return

    # 2. 准备映射表
    with open(CONFIG["gene_voc_path"], 'r') as f:
        symbol_to_id = json.load(f)
    id_to_symbol = {v: k for k, v in symbol_to_id.items()}

    # 3. 任务分配
    n = CONFIG["num_processes"]
    task_chunks = [all_tasks[i::n] for i in range(n)]
    
    # 4. 启动多进程并监控
    print(f"🚀 总任务数: {len(all_tasks)} | 进程数: {n} | GPU: {CONFIG['available_gpus']}")
    
    processes = []
    try:
        for i in range(n):
            gpu_id = CONFIG["available_gpus"][i % len(CONFIG["available_gpus"])]
            p = mp.Process(target=worker, args=(i, task_chunks[i], gpu_id, id_to_symbol))
            p.daemon = True # 设置为守护进程，主进程退出时子进程尝试关闭
            p.start()
            processes.append(p)

        # 阻塞主进程，直到所有子进程完成
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("\n[STOP] 接收到停止指令 (Ctrl+C)，正在清理进程并释放显存...")
        for p in processes:
            if p.is_alive():
                p.terminate() # 强制杀掉子进程
                p.join()      # 回收资源
        print("✅ 所有子进程已安全停止。")
        os._exit(0) # 彻底退出主程序

    print("✅ 所有批量任务处理完成！")

if __name__ == "__main__":
    # 使用 spawn 模式确保多卡环境下 CUDA 状态清洁
    mp.set_start_method('spawn', force=True)
    main()