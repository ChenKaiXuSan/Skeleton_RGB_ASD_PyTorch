#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/SAM3Dbody/main_multi_gpu_process.py
Project: /workspace/code/SAM3Dbody
Created Date: Monday January 26th 2026
Author: Kaixu Chen
-----
Comment:
根据多GPU并行处理SAM-3D-Body推理任务。

Have a good code time :)
-----
Last Modified: Monday January 26th 2026 5:12:10 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import os
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

# 假设这些是从你的其他模块导入的
from .infer import process_frame_list
from .load import load_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# 核心处理逻辑：处理单个人的数据
# ---------------------------------------------------------------------
def process_single_video(
    video_path: Path,
    source_root: Path,
    out_root: Path,
    infer_root: Path,
    cfg: DictConfig,
    start_mid_end_dict: Optional[dict] = None,
):
    """处理单个视频文件"""
    rel_video = video_path.relative_to(source_root)
    disease_name = rel_video.parts[0] if len(rel_video.parts) >= 2 else "root"
    video_name = video_path.stem

    # --- 1. Person専用のログ設定 ---
    log_dir = out_root / "person_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    video_log_file = log_dir / f"{disease_name}_{video_name}.log"

    # 新しいハンドラを作成
    handler = logging.FileHandler(video_log_file, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger(
        f"{disease_name}_{video_name}"
    )  # このVideo専用のロガーを取得
    logger.addHandler(handler)
    # logger.propagate = False  # 親（Root）ロガーにログを流さない（混ざるのを防ぐ）

    logger.info(
        f"==== Starting Process for Disease: {disease_name}, Video: {video_name} ===="
    )

    view_frames: Dict[str, List[np.ndarray]] = load_data({"video": video_path.resolve()})

    rel_parent = Path(disease_name)

    for view, frames in view_frames.items():
        logger.info(f"  视角 {view} 处理了 {len(frames)} 帧数据。")
        _out_root = out_root / rel_parent / video_name / view
        _out_root.mkdir(parents=True, exist_ok=True)
        _infer_root = infer_root / rel_parent / video_name / view
        _infer_root.mkdir(parents=True, exist_ok=True)

        process_frame_list(
            frame_list=frames,
            out_dir=_out_root,
            inference_output_path=_infer_root,
            start_mid_end_dict=start_mid_end_dict,
            cfg=cfg,
        )


# ---------------------------------------------------------------------
# GPU Worker：进程执行函数
# ---------------------------------------------------------------------
def gpu_worker(
    gpu_id: int,
    video_paths: List[Path],
    source_root: Path,
    out_root: Path,
    infer_root: Path,
    cfg_dict: dict,
):
    """
    每个进程的入口：设置环境变量，并处理分配的任务列表
    """
    # 1. 隔离 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cfg_dict["infer"]["gpu"] = 0  # 因为上面已经隔离了 GPU，所以这里设为 0

    # 2. 将字典转回 Hydra 配置（多进程传递对象时，转为字典更安全）
    cfg = OmegaConf.create(cfg_dict)

    logger.info(f"🟢 GPU {gpu_id} 进程启动，待处理任务数: {len(video_paths)}")

    for video_path in video_paths:
    # 从后面往前推导 
    # for video_path in reversed(video_paths):
        try:
            process_single_video(video_path, source_root, out_root, infer_root, cfg)
        except Exception as e:
            logger.error(f"❌ GPU {gpu_id} 处理 {video_path.name} 时出错: {e}")

    logger.info(f"🏁 GPU {gpu_id} 所有任务处理完毕")


# ---------------------------------------------------------------------
# Main 入口
# ---------------------------------------------------------------------
@hydra.main(config_path="../configs", config_name="sam3d_body", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1. 経路準備
    out_root = Path(cfg.paths.log_path).resolve()
    infer_root = Path(cfg.paths.result_output_path).resolve()
    source_root = Path(cfg.paths.video_path).resolve()

    # --- 1. 读取 video 下的视频任务 ---
    all_video_tasks: List[Path] = []
    video_patterns = ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV")
    target_diseases = set(cfg.infer.get("disease_list", ["all"]))

    # 1.1 根目录视频（兼容旧扁平结构）
    for pattern in video_patterns:
        all_video_tasks.extend(sorted(source_root.glob(pattern)))

    # 1.2 疾病子目录视频（当前结构：video/ASD|DHS|LCS|HipOA/*.mp4）
    for disease_dir in sorted(source_root.iterdir()):
        if not disease_dir.is_dir():
            continue

        if disease_dir.name not in target_diseases and "all" not in target_diseases:
            continue

        for pattern in video_patterns:
            all_video_tasks.extend(sorted(disease_dir.glob(pattern)))

    all_video_tasks = sorted(set(all_video_tasks))

    if not all_video_tasks:
        logger.error(f"未找到视频任务，请检查目录: {source_root}")
        return

    # --- 2. 並列処理の実行 ---
    gpu_ids = cfg.infer.get("gpu", [0, 1])
    workers_per_gpu = cfg.infer.get("workers_per_gpu", 2)

    total_workers = len(gpu_ids) * workers_per_gpu
    chunks = np.array_split(all_video_tasks, total_workers)

    logger.info(f"使用 GPU: {gpu_ids} (各 {workers_per_gpu} ワーカー)")
    logger.info(f"総プロセス数: {total_workers}")
    logger.info(f"总视频任务数: {len(all_video_tasks)}")
    logger.info(f"疾病过滤: {sorted(target_diseases)}")

    # 3. 启动并行进程
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mp.set_start_method("spawn", force=True)

    processes = []
    for i, gpu_id in enumerate(np.repeat(gpu_ids, workers_per_gpu)):
        video_list = chunks[i].tolist()
        if not video_list:
            continue

        logger.info(f"  - Worker {i} (GPU {gpu_id}) 分配任务数: {len(video_list)}")

        p = mp.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                video_list,
                source_root,
                out_root,
                infer_root,
                cfg_dict,
            ),
        )
        p.start()
        processes.append(p)

    # 4. 等待所有进程完成
    for p in processes:
        p.join()

    logger.info("🎉 [SUCCESS] 所有 GPU 任务已圆满完成！")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
