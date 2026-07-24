#!/usr/bin/env python3
"""DiffDock wrapper — 供 ToolManager 通过 diffdock conda env 调用。

⚠️ 必须在 DiffDock 目录下运行（SO(2)/SO(3) 预计算缓存文件依赖）


用法:
    python scripts/diffdock_wrapper.py <protein.pdb> <ligand.sdf|SMILES> <out_dir> [--samples N]

输出:
    out_dir/rank1.sdf          — 最佳置信度姿态
    out_dir/rank1_confidence*.sdf — 各排名姿态
    out_dir/result.json        — 摘要 JSON
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, RemoveAllHs, SDWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# 添加 DiffDock 源代码到路径
DIFFDOCK_HOME = Path(os.environ.get("DIFFDOCK_HOME", "/users_home/wangpengzheng/software/DiffDock"))
sys.path.insert(0, str(DIFFDOCK_HOME))

from datasets.process_mols import write_mol_with_coords
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.inference_utils import InferenceDataset
from utils.sampling import randomize_position, sampling
from utils.utils import get_model

RDLogger.DisableLog("rdApp.*")


def run_diffdock(
    protein_path: Path,
    ligand_desc: str,  # SMILES string or SDF path
    out_dir: Path,
    model_dir: Path | None = None,
    confidence_model_dir: Path | None = None,
    samples_per_complex: int = 10,
    inference_steps: int = 20,
    device: str = "cuda",
) -> dict:
    """Run DiffDock inference and return results summary."""

    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse config
    if model_dir is None:
        model_dir = DIFFDOCK_HOME / "workdir" / "v1.1" / "score_model"
        confidence_model_dir = DIFFDOCK_HOME / "workdir" / "v1.1" / "confidence_model"

    with open(model_dir / "model_parameters.yml") as f:
        score_model_args = yaml.full_load(f)
        # 兼容 Namespace 用法
        from argparse import Namespace
        score_model_args = Namespace(**score_model_args)

    if confidence_model_dir and confidence_model_dir.exists():
        with open(confidence_model_dir / "model_parameters.yml") as f:
            confidence_args = yaml.full_load(f)
            confidence_args = Namespace(**confidence_args)
    else:
        confidence_model_dir = None
        confidence_args = None

    # 推理设备
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # 构建测试数据集
    complex_name = "diffdock_complex"
    test_dataset = InferenceDataset(
        out_dir=str(out_dir),
        complex_names=[complex_name],
        protein_files=[str(protein_path)],
        ligand_descriptions=[ligand_desc],
        protein_sequences=[None],
        lm_embeddings=True,
        receptor_radius=score_model_args.receptor_radius,
        remove_hs=score_model_args.remove_hs,
        c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
        all_atoms=score_model_args.all_atoms,
        atom_radius=score_model_args.atom_radius,
        atom_max_neighbors=score_model_args.atom_max_neighbors,
        knn_only_graph=False,
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # 准备置信度数据集（如果需要）
    if confidence_model_dir is not None and confidence_args is not None:
        if hasattr(confidence_args, 'use_original_model_cache') and not confidence_args.use_original_model_cache:
            confidence_test_dataset = InferenceDataset(
                out_dir=str(out_dir),
                complex_names=[complex_name],
                protein_files=[str(protein_path)],
                ligand_descriptions=[ligand_desc],
                protein_sequences=[None],
                lm_embeddings=True,
                receptor_radius=confidence_args.receptor_radius,
                remove_hs=confidence_args.remove_hs,
                c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                all_atoms=confidence_args.all_atoms,
                atom_radius=confidence_args.atom_radius,
                atom_max_neighbors=confidence_args.atom_max_neighbors,
                precomputed_lm_embeddings=test_dataset.lm_embeddings,
                knn_only_graph=False,
            )
        else:
            confidence_test_dataset = None
    else:
        confidence_test_dataset = None

    # 加载模型
    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, old=False)
    state_dict = torch.load(str(model_dir / "best_ema_inference_epoch_model.pt"), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if confidence_model_dir is not None and confidence_args is not None:
        confidence_model = get_model(
            confidence_args, device, t_to_sigma=t_to_sigma,
            no_parallel=True, confidence_mode=True, old=True,
        )
        state_dict = torch.load(str(confidence_model_dir / "best_model_epoch75.pt"), map_location="cpu")
        confidence_model.load_state_dict(state_dict, strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()
    else:
        confidence_model = None
        confidence_args = None

    # 运行推理
    tr_schedule = get_t_schedule(inference_steps=inference_steps, sigma_schedule="expbeta")
    N = samples_per_complex

    results = {"protein": str(protein_path), "ligand": ligand_desc, "poses": []}

    for orig_complex_graph in tqdm(test_loader, desc="DiffDock"):
        if not orig_complex_graph.success[0]:
            results["status"] = "failed"
            results["error"] = "Dataset preprocessing failed"
            return results

        try:
            if confidence_test_dataset is not None:
                confidence_complex_graph = confidence_test_dataset[0]
                if not confidence_complex_graph.success:
                    continue
                confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
            else:
                confidence_data_list = None

            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
            randomize_position(
                data_list, score_model_args.no_torsion, False,
                score_model_args.tr_sigma_max,
                initial_noise_std_proportion=-1.0,
                choose_residue=False,
            )

            lig = orig_complex_graph.mol[0]

            # 反向扩散
            data_list, confidence = sampling(
                data_list=data_list, model=model,
                inference_steps=inference_steps,
                tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule,
                device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                visualization_list=None, confidence_model=confidence_model,
                confidence_data_list=confidence_data_list, confidence_model_args=confidence_args,
                batch_size=min(N, 10), no_final_step_noise=True,
                temp_sampling=[1.17, 2.06, 7.04],
                temp_psi=[0.73, 0.90, 0.59],
                temp_sigma_data=[0.93, 0.75, 0.69],
            )

            # 提取坐标
            ligand_pos = np.asarray([
                complex_graph["ligand"].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy()
                for complex_graph in data_list
            ])

            # 按置信度排序
            if confidence is not None:
                confidence = confidence.cpu().numpy()
                if confidence.ndim > 1:
                    confidence = confidence[:, 0]
                re_order = np.argsort(confidence)[::-1]
                confidence = confidence[re_order]
                ligand_pos = ligand_pos[re_order]

            # 保存结果
            for rank, pos in enumerate(ligand_pos):
                mol_pred = copy.deepcopy(lig)
                if score_model_args.remove_hs:
                    mol_pred = RemoveAllHs(mol_pred)
                conf_val = float(confidence[rank]) if confidence is not None else 0.0
                write_mol_with_coords(mol_pred, pos, str(out_dir / f"rank{rank+1}.sdf"))
                write_mol_with_coords(mol_pred, pos, str(out_dir / f"rank{rank+1}_confidence{conf_val:.2f}.sdf"))
                results["poses"].append({
                    "rank": rank + 1,
                    "confidence": round(conf_val, 4),
                    "sdf": str(out_dir / f"rank{rank+1}.sdf"),
                })

            results["status"] = "ok"
            results["top_confidence"] = float(confidence[0]) if confidence is not None else None

        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            import traceback
            results["traceback"] = traceback.format_exc()

    return results


def main():
    parser = argparse.ArgumentParser(description="DiffDock wrapper")
    parser.add_argument("protein", type=str, help="Path to protein PDB")
    parser.add_argument("ligand", type=str, help="SMILES string or path to ligand SDF")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("--samples", type=int, default=10, help="Samples per complex (default: 10)")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps (default: 20)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    result = run_diffdock(
        protein_path=Path(args.protein),
        ligand_desc=args.ligand,
        out_dir=Path(args.out_dir),
        samples_per_complex=args.samples,
        inference_steps=args.steps,
        device=args.device,
    )

    result_path = Path(args.out_dir) / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    if result.get("status") == "ok":
        print(json.dumps({"status": "ok", "poses": len(result["poses"]), "top_confidence": result.get("top_confidence")}))
    else:
        print(json.dumps({"status": "failed", "error": result.get("error", "unknown")}))
        sys.exit(1)


if __name__ == "__main__":
    main()
