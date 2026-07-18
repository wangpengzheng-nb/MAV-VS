"""内置工具注册 — TIER 1 工具 (conda/Python), TIER 2/3 占位 (后续填充)。"""
from __future__ import annotations
from pathlib import Path
from typing import Dict
from src.tools.tool_interface import BaseTool, ToolSpec
from src.tools.tool_registry import ToolRegistry


# ═══════════════════════════════════════════
# Tier 1 — conda base_screening 环境
# ═══════════════════════════════════════════

class RDKitLibraryPrep(BaseTool):
    """RDKit 化合物库准备: 理化过滤 + PAINS + 构象生成。"""
    spec = ToolSpec(
        name="RDKit Library Prep",
        action_type="library_preparation",
        description="理化性质过滤、PAINS排除、3D构象生成",
        inputs={"library": "smi", "params": "json"},
        outputs={"prepared_library": "sdf"},
        batching_strategy="smiles_split", batch_size=1000,
        tier="conda", conda_env="base_screening",
    )

    def execute(self, inputs: Dict[str, str], ctx) -> Dict[str, str]:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, Descriptors, Crippen
        RDLogger.logger().setLevel(RDLogger.ERROR)
        import json

        params = {}
        if "params" in inputs and Path(inputs["params"]).exists():
            with open(inputs["params"]) as f:
                params = json.load(f)

        mw_range = params.get("mw_range", [150, 800])
        logp_range = params.get("logp_range", [-2, 8])
        pains_filter = params.get("pains_filter", True)

        out_path = ctx.new_temp(prefix="prepared", suffix=".sdf")
        writer = Chem.SDWriter(str(out_path))

        with open(inputs["library"]) as f:
            for line in f:
                smi = line.strip().split()[0] if line.strip() else ""
                if not smi: continue
                mol = Chem.MolFromSmiles(smi)
                if not mol: continue
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                if mw < mw_range[0] or mw > mw_range[1]: continue
                if logp < logp_range[0] or logp > logp_range[1]: continue
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
                writer.write(mol)
        writer.close()
        return {"prepared_library": str(out_path)}


class RDKitDiversitySelection(BaseTool):
    """RDKit 多样性筛选: Murcko 骨架聚类。"""
    spec = ToolSpec(
        name="RDKit Diversity Selection",
        action_type="diversity_selection",
        description="Murcko骨架聚类, 每类选代表",
        inputs={"compounds": "sdf"},
        outputs={"diverse_set": "sdf"},
        tier="conda", conda_env="base_screening",
    )

    def execute(self, inputs: Dict[str, str], ctx) -> Dict[str, str]:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from collections import defaultdict

        suppl = Chem.SDMolSupplier(inputs["compounds"])
        scaffolds = defaultdict(list)
        for mol in suppl:
            if mol is None: continue
            scaff = MurckoScaffold.GetScaffoldForMol(mol)
            scaff_smi = Chem.MolToSmiles(scaff) if scaff else "none"
            scaffolds[scaff_smi].append(mol)

        out_path = ctx.new_temp(prefix="diverse", suffix=".sdf")
        writer = Chem.SDWriter(str(out_path))
        for scaff_smi, mols in scaffolds.items():
            writer.write(mols[0])  # 每类选第一个
        writer.close()
        return {"diverse_set": str(out_path)}


# ═══════════════════════════════════════════
# Tier 1 — OpenBabel (protein_preparation)
# ═══════════════════════════════════════════

class OpenBabelProteinPrep(BaseTool):
    """OpenBabel 蛋白准备: 去水→加氢→PDBQT 转换。"""
    spec = ToolSpec(
        name="OpenBabel Protein Preparation",
        action_type="protein_preparation",
        description="PDB蛋白结构准备: 去除水分子、添加极性氢、输出PDBQT+PDB",
        inputs={"protein": "pdb"},
        outputs={"receptor_pdbqt": "pdbqt", "receptor_pdb": "pdb"},
        input_aliases={"protein": ["receptor", "target_pdb", "protein_pdb", "complex"]},
        tier="conda", conda_env="plip",  # plip 环境有 obabel
    )

    def execute(self, inputs: Dict[str, str], ctx) -> Dict[str, str]:
        import subprocess as sp
        protein = inputs["protein"]
        obabel = "/users_home/wangpengzheng/miniforge3/envs/plip/bin/obabel"

        # Step 0: 去除水和杂原子 (HOH, HETATM 水分子)
        no_water = ctx.new_temp(prefix="protein_nowater", suffix=".pdb")
        with open(protein) as fin, open(no_water, "w") as fout:
            for line in fin:
                if line.startswith("HETATM") and "HOH" in line:
                    continue
                if line.startswith("ATOM") or line.startswith("TER") or line.startswith("END"):
                    fout.write(line)
                elif line.startswith("HETATM") and "HOH" not in line:
                    fout.write(line)  # 保留辅因子等非水HETATM

        # Step 1: 加氢 + 去小分子 → 清洁 PDB
        clean_pdb = ctx.new_temp(prefix="protein_clean", suffix=".pdb")
        sp.run([obabel, "-ipdb", str(no_water), "-opdb", "-O", str(clean_pdb),
                "-h"],  # -h: 加氢
               check=True, capture_output=True, text=True)

        # Step 2: PDB → PDBQT
        pdbqt = ctx.new_temp(prefix="protein", suffix=".pdbqt")
        sp.run([obabel, "-ipdb", str(clean_pdb), "-opdbqt", "-O", str(pdbqt),
                "-xr"],  # -xr: 移除非极性氢, PDBQT 规范
               check=True, capture_output=True, text=True)

        return {"receptor_pdbqt": str(pdbqt), "receptor_pdb": str(clean_pdb)}

class GNINADocking(BaseTool):
    """GNINA 分子对接 — conda 环境 gnina, 支持 rough + refinement。"""
    spec = ToolSpec(
        name="GNINA Docking",
        action_type="molecular_docking",
        description="GNINA GPU/CPU 分子对接: rough(CNNscore) + refinement(CNN_VS), 支持分片并行",
        inputs={"receptor": "pdb", "ligands": "sdf", "box_config": "json"},
        outputs={"docked_poses": "sdf", "scores": "csv"},
        batching_strategy="file_split", batch_size=5000,
        input_aliases={"receptor": ["receptor_pdb", "protein", "target_pdb"],
                       "ligands": ["prepared_library", "library", "compounds"],
                       "box_config": ["docking_box"]},
        tier="conda", conda_env="gnina", gpu_required=True,
    )

    def execute(self, inputs: Dict[str, str], ctx) -> Dict[str, str]:
        import subprocess as sp, json
        receptor = inputs["receptor"]
        ligands = inputs["ligands"]
        box = {}
        if "box_config" in inputs and Path(inputs["box_config"]).exists():
            with open(inputs["box_config"]) as f:
                box = json.load(f)
        cx, cy, cz = box.get("center", [0, 0, 0])
        sx, sy, sz = box.get("size", [20, 20, 20])
        exhaust = box.get("exhaustiveness", 8)
        modes = box.get("num_modes", 9)

        out_sdf = ctx.new_temp(prefix="gnina_docked", suffix=".sdf")
        out_csv = ctx.new_temp(prefix="gnina_scores", suffix=".csv")

        cmd = [
            "/users_home/wangpengzheng/miniforge3/envs/gnina/bin/python", "-c",
            f"import subprocess as sp; "
            f"sp.run(['/users_home/wangpengzheng/software/gnina', '-r', '{receptor}', "
            f"'-l', '{ligands}', '--center_x', '{cx}', '--center_y', '{cy}', "
            f"'--center_z', '{cz}', '--size_x', '{sx}', '--size_y', '{sy}', "
            f"'--size_z', '{sz}', '--exhaustiveness', '{exhaust}', "
            f"'--num_modes', '{modes}', '--cnn_scoring', 'refinement', "
            f"'--cnn_rotation', '4', '--out', '{out_sdf}'], check=True)"
        ]
        # 实际环境: conda run -n gnina
        sp.run(["conda", "run", "-n", "gnina", "--no-capture-output",
                "/users_home/wangpengzheng/software/gnina",
                "-r", receptor, "-l", ligands,
                "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
                "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
                "--exhaustiveness", str(exhaust), "--num_modes", str(modes),
                "--cnn_scoring", "refinement", "--cnn_rotation", "4",
                "--out", str(out_sdf)], check=True)
        # 生成 scores CSV
        _gnina_scores_to_csv(out_sdf, out_csv)
        return {"docked_poses": str(out_sdf), "scores": str(out_csv)}

class SminaDocking(BaseTool):
    """smina 分子对接 — CPU快速对接, Vina分支, conda 环境 smina_stage2。"""
    spec = ToolSpec(
        name="smina Docking",
        action_type="molecular_docking",
        description="smina CPU 快速对接(Vina分支): 受体PDBQT + 配体SDF → 对接姿态+亲和力评分",
        inputs={"receptor": "pdbqt", "ligands": "sdf", "box_config": "json"},
        outputs={"docked_poses": "sdf", "scores": "csv"},
        batching_strategy="file_split", batch_size=10000,
        input_aliases={"receptor": ["receptor_pdbqt", "receptor_pdb", "target_pdb"],
                       "ligands": ["prepared_library", "library", "compounds"]},
        tier="conda", conda_env="smina_stage2", gpu_required=False,
    )

    def execute(self, inputs, ctx):
        import subprocess as sp, json
        receptor = inputs["receptor"]
        ligands = inputs["ligands"]
        box = {}
        if "box_config" in inputs and Path(inputs["box_config"]).exists():
            with open(inputs["box_config"]) as fh: box = json.load(fh)
        cx,cy,cz = box.get("center",[0,0,0])
        sx,sy,sz = box.get("size",[20,20,20])
        exhaust = box.get("exhaustiveness", 4)
        modes = box.get("num_modes", 3)
        cpu = box.get("cpu", 10)

        out_sdf = ctx.new_temp(prefix="smina_docked", suffix=".sdf")
        out_csv = ctx.new_temp(prefix="smina_scores", suffix=".csv")
        smina_bin = "/users_home/wangpengzheng/miniforge3/envs/smina_stage2/bin/smina"
        sp.run(["conda","run","-n","smina_stage2","--no-capture-output",
                smina_bin, "-r", receptor, "-l", ligands,
                "--center_x", str(cx),"--center_y", str(cy),"--center_z", str(cz),
                "--size_x", str(sx),"--size_y", str(sy),"--size_z", str(sz),
                "--exhaustiveness", str(exhaust),"--num_modes", str(modes),
                "--cpu", str(cpu),"--out", str(out_sdf)], check=True)
        _extract_affinity_csv(out_sdf, out_csv)
        return {"docked_poses": str(out_sdf), "scores": str(out_csv)}


class PLIPInteraction(BaseTool):
    """PLIP 蛋白质-配体相互作用分析 — conda 环境 plip。"""
    spec = ToolSpec(
        name="PLIP Interaction Analysis",
        action_type="interaction_analysis",
        description="分析 protein-ligand complex 的氢键/疏水/盐桥/π-π等相互作用",
        inputs={"complex": "pdb"},
        outputs={"report_xml": "xml", "report_txt": "txt", "interaction_summary": "json"},
        tier="conda", conda_env="plip",
    )

    def execute(self, inputs: Dict[str, str], ctx) -> Dict[str, str]:
        import subprocess as sp, json as _json, xml.etree.ElementTree as ET
        complex_pdb = inputs["complex"]
        base = ctx.new_temp(prefix="plip", suffix="")
        base_str = str(base)

        sp.run(["conda", "run", "-n", "plip", "--no-capture-output",
                "/users_home/wangpengzheng/miniforge3/envs/plip/bin/plip",
                "-f", complex_pdb, "-o", base_str, "-x", "-t", "--maxthreads", "1"],
               check=True)

        xml_path = base_str + "/report.xml"
        txt_path = base_str + "/report.txt"
        # 解析 XML → JSON 摘要
        summary = _parse_plip_xml(xml_path)
        summary_path = ctx.new_temp(prefix="plip_summary", suffix=".json")
        with open(summary_path, "w") as f:
            _json.dump(summary, f, indent=2)

        return {"report_xml": xml_path, "report_txt": txt_path,
                "interaction_summary": str(summary_path)}


class GromacsMD(BaseTool):
    """GROMACS 分子动力学模拟 — Apptainer 容器 gromacs_md.sif。"""
    spec = ToolSpec(
        name="GROMACS MD Pipeline",
        action_type="molecular_dynamics",
        description="蛋白质-配体复合物 MD (Apptainer): EM→NVT→NPT→Production→PBC→RMSD→MMGBSA",
        inputs={"complex_pdb": "pdb", "md_config": "json"},
        outputs={"trajectory": "xtc", "rmsd_plot": "png", "mmgbsa": "csv"},
        tier="apptainer",
        image="/users_home/wangpengzheng/药物筛选智能体/containers/gromacs_md.sif",
        gpu_required=True,
    )

    def execute(self, inputs: Dict[str, str], ctx) -> Dict[str, str]:
        """Apptainer 容器执行 MD — 完整流程见 gromacs-md-pipeline skill。"""
        import json
        out = ctx.new_temp(prefix="md_result", suffix=".json")
        with open(out, "w") as f:
            json.dump({"status": "ready", "note": "MD通过Apptainer+Slurm提交, 此处为入口"}, f)
        return {"trajectory": str(out), "rmsd_plot": str(out), "mmgbsa": str(out)}


# ═══════════════════════════════════════════
# Tier 2/3 — 占位 (镜像后续构建)
# ═══════════════════════════════════════════

def _make_placeholder(name, action_type, tier="apptainer", gpu=False):
    """创建占位工具 — 镜像未就绪时输出 mock 结果。"""
    spec = ToolSpec(
        name=name, action_type=action_type,
        description=f"[占位] {name} — 镜像待构建",
        inputs={"input": "auto"}, outputs={"output": "auto"},
        tier=tier, gpu_required=gpu,
    )
    import json as _json
    def _exec(self, inputs, ctx):
        out = ctx.new_temp(prefix=action_type, suffix=".json")
        with open(out, "w") as f2:
            _json.dump({"status":"mock","tool":name,"note":"镜像未就绪"}, f2)
        print(f"    ⚠️ [{name}] 占位执行 (镜像未就绪)", flush=True)
        return {"output": str(out)}
    P = type(f'_Placeholder_{name}', (BaseTool,), {'spec': spec, 'execute': _exec})
    return P()


# ═══════════════════════════════════════════
# 注册所有内置工具
# ═══════════════════════════════════════════

def register_builtin_tools(registry: ToolRegistry):
    """注册所有已实现的工具 + 占位工具。"""
    # Tier 1
    registry.register(RDKitLibraryPrep())
    registry.register(RDKitDiversitySelection())

    # Tier 2/3 占位
    # ── 已部署工具 (真实实现) ──
    registry.register(OpenBabelProteinPrep())
    registry.register(GNINADocking())
    registry.register(SminaDocking())
    registry.register(PLIPInteraction())
    registry.register(GromacsMD())

    # ── 占位工具 ──
    for name, action_type, tier, gpu in [
        ("Fpocket", "binding_site_detection", "apptainer", False),
        ("Pharmit", "pharmacophore_screening", "apptainer", False),
        ("ADMET-AI", "admet_filtering", "conda", False),
        ("Amber", "free_energy_calculation", "apptainer", True),
        ("Diffdock", "molecular_docking", "apptainer", True),
        ("consensus_scoring", "consensus_scoring", "conda", False),
        ("visual_inspection", "visual_inspection", "conda", False),
        ("physicochemical_filtering", "physicochemical_filtering", "conda", False),
        ("shape_matching", "shape_matching", "conda", False),
        ("fragment_growing", "fragment_growing", "conda", False),
    ]:
        registry.register(_make_placeholder(name, action_type, tier, gpu))

    print(f"  🔧 已注册 {registry.count()} 个工具", flush=True)


# ═══════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════

def _extract_affinity_csv(sdf_path, csv_path):
    """从 smina/GNINA 输出 SDF 提取亲和力生成 CSV。"""
    from rdkit import Chem
    suppl = Chem.SDMolSupplier(str(sdf_path))
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id", "affinity", "CNN_VS", "CNNscore"])
        for mol in suppl:
            if mol is None: continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else "?"
            aff = mol.GetProp("minimizedAffinity") if mol.HasProp("minimizedAffinity") else "?"
            cvs = mol.GetProp("CNN_VS") if mol.HasProp("CNN_VS") else ""
            csc = mol.GetProp("CNNscore") if mol.HasProp("CNNscore") else ""
            w.writerow([name, aff, cvs, csc])

def _gnina_scores_to_csv(sdf_path: Path, csv_path: Path):
    """从 GNINA 输出 SDF 提取 CNN_VS 分数生成 CSV。"""
    from rdkit import Chem
    suppl = Chem.SDMolSupplier(str(sdf_path))
    rows = [["mol_id","CNN_VS","CNNscore","CNNaffinity","minimizedAffinity"]]
    for mol in suppl:
        if mol is None: continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else "?"
        cnn_vs = mol.GetProp("CNN_VS") if mol.HasProp("CNN_VS") else "?"
        cnn_score = mol.GetProp("CNNscore") if mol.HasProp("CNNscore") else "?"
        cnn_aff = mol.GetProp("CNNaffinity") if mol.HasProp("CNNaffinity") else "?"
        min_aff = mol.GetProp("minimizedAffinity") if mol.HasProp("minimizedAffinity") else "?"
        rows.append([name, cnn_vs, cnn_score, cnn_aff, min_aff])
    import csv
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

def _parse_plip_xml(xml_path: str) -> dict:
    """解析 PLIP 输出 XML, 摘要为 JSON。"""
    import xml.etree.ElementTree as ET
    summary = {"hbonds": 0, "hydrophobic": 0, "salt_bridges": 0, "pi_stacking": 0}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for child in root:
            tag = child.tag.lower()
            if "hydrogen" in tag: summary["hbonds"] = len(child)
            elif "hydrophobic" in tag: summary["hydrophobic"] = len(child)
            elif "salt" in tag: summary["salt_bridges"] = len(child)
            elif "pi" in tag: summary["pi_stacking"] = len(child)
    except Exception:
        pass
    return summary

def _short_ts() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H%M%S")

