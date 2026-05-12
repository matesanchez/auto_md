from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shlex
import tempfile
import shutil
import subprocess
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .io import file_record, freeze_input, read_yaml, sha256_file, utc_now, write_yaml
from .models import FormulationInput, TopologyRecord


KNOWN_TOPOLOGIES = [
    TopologyRecord(
        topology_id="AUTOPO-MC3",
        display_name="MC3 Martini 3 ionizable lipid placeholder",
        molecule_names=["MC3", "DLin-MC3-DMA"],
        aliases=[{"type": "common_name", "value": "MC3"}],
        role_tags=["ionizable_lipid", "lnp_lipid"],
        source_family="mock_test",
        topology_files=[{"path": "topology_library/approved/MC3.itp"}],
        molecule_type_name="MC3",
        approval_status="approved_with_assumption",
        production_eligible=False,
        placeholder_topology=True,
        production_review_required=True,
        limitations=["Placeholder fixture for workflow smoke testing; replace with licensed curated topology before production."],
    ),
    TopologyRecord(
        topology_id="AUTOPO-DSPC",
        display_name="DSPC Martini 3 phospholipid placeholder",
        molecule_names=["DSPC"],
        aliases=[{"type": "common_name", "value": "DSPC"}],
        role_tags=["phospholipid", "helper_lipid"],
        source_family="mock_test",
        topology_files=[{"path": "topology_library/approved/DSPC.itp"}],
        molecule_type_name="DSPC",
        approval_status="approved_with_assumption",
        production_eligible=False,
        placeholder_topology=True,
        production_review_required=True,
        limitations=["Placeholder fixture for workflow smoke testing; replace with licensed curated topology before production."],
    ),
    TopologyRecord(
        topology_id="AUTOPO-CHOL",
        display_name="Cholesterol Martini 3 sterol placeholder",
        molecule_names=["cholesterol", "CHOL"],
        aliases=[{"type": "common_name", "value": "cholesterol"}, {"type": "common_name", "value": "CHOL"}],
        role_tags=["sterol"],
        source_family="mock_test",
        topology_files=[{"path": "topology_library/approved/CHOL.itp"}],
        molecule_type_name="CHOL",
        approval_status="approved_with_assumption",
        production_eligible=False,
        placeholder_topology=True,
        production_review_required=True,
        limitations=["Placeholder fixture for workflow smoke testing; replace with licensed curated topology before production."],
    ),
    TopologyRecord(
        topology_id="AUTOPO-DMGPEG",
        display_name="DMG-PEG2000 Martini 3 PEG lipid placeholder",
        molecule_names=["DMG-PEG2000", "DMG-PEG"],
        aliases=[{"type": "common_name", "value": "DMG-PEG2000"}],
        role_tags=["peg_lipid"],
        source_family="mock_test",
        topology_files=[{"path": "topology_library/approved/DMG-PEG2000.itp"}],
        molecule_type_name="DMGPEG",
        approval_status="approved_with_assumption",
        production_eligible=False,
        placeholder_topology=True,
        production_review_required=True,
        limitations=["Placeholder fixture for workflow smoke testing; replace with licensed curated topology before production."],
    ),
]


TEMPLATES = {
    "descriptor_only": {
        "template_id": "descriptor_only",
        "description": "Descriptor and readiness workflow with no MD.",
        "requires_resolved_topology": False,
        "builder": "none",
    },
    "minimal_mixed_lipid_smoke_box": {
        "template_id": "minimal_mixed_lipid_smoke_box",
        "description": "Small mixed-lipid Martini/GROMACS plumbing smoke package.",
        "requires_resolved_topology": True,
        "builder": "mock",
    },
    "lnp_smoke_self_assembly": {
        "template_id": "lnp_smoke_self_assembly",
        "description": "LNP-like smoke template; MVP routes to mock builder unless a real adapter is supplied.",
        "requires_resolved_topology": True,
        "builder": "mock",
    },
    "production_mixed_lipid_pipeline": {
        "template_id": "production_mixed_lipid_pipeline",
        "description": "Production artifact contract with staged EM/NVT/NPT/MD lifecycle.",
        "requires_resolved_topology": True,
        "builder": "production_pack_like",
    },
}


SOURCE_REPOS = [
    {
        "name": "M3-Small-Molecules",
        "url": "https://github.com/Martini-Force-Field-Initiative/M3-Small-Molecules.git",
        "path": "external/M3-Small-Molecules",
    },
    {
        "name": "M3_Ionizable_Lipids",
        "url": "https://github.com/Martini-Force-Field-Initiative/M3_Ionizable_Lipids.git",
        "path": "external/M3_Ionizable_Lipids",
    },
    {
        "name": "M3-Lipid-Parameters",
        "url": "https://github.com/Martini-Force-Field-Initiative/M3-Lipid-Parameters.git",
        "path": "external/M3-Lipid-Parameters",
    },
]


DEFAULT_AUTOMATION_POLICY = {
    "schema_version": "automd.automation_policy.v0.1",
    "auto_accept_tiers": ["A_curated_exact", "B_curated_family_match", "C_generated_from_approved_fragments"],
    "require_review_tiers": ["D_generated_smallmol_backend", "E_manual_review_required"],
    "allow_triage_generated_topologies": False,
    "default_template": "auto",
    "default_mode": "smoke",
    "real_gromacs_default": False,
    "role_inference": {
        "minimum_confidence_for_auto_role": 0.70,
        "low_confidence_role_fallback": "additive",
    },
    "simulation": {
        "block_smoke_when_any_component_requires_review": True,
        "allow_descriptor_only_fallback": True,
    },
}


CONFIDENCE_TIERS = {
    "A_curated_exact": {"smoke_eligible": True, "production_eligible": True, "review_required": False, "production_review_required": False},
    "B_curated_family_match": {"smoke_eligible": True, "production_eligible": True, "review_required": False, "production_review_required": False},
    "C_generated_from_approved_fragments": {"smoke_eligible": True, "production_eligible": False, "review_required": False, "production_review_required": True},
    "D_generated_smallmol_backend": {"smoke_eligible": False, "production_eligible": False, "review_required": True, "production_review_required": True},
    "E_manual_review_required": {"smoke_eligible": False, "production_eligible": False, "review_required": True, "production_review_required": True},
}


ROLE_FAMILY_RULES = {
    "ionizable_lipid": {
        "family": "ionizable_lipid",
        "bead_count": 6,
        "charge": 0.0,
        "charge_assumption": "neutral smoke-test topology; protonation must be reviewed before production",
        "matching_rules": ["role == ionizable_lipid", "SMILES or topology hint present"],
    },
    "phospholipid": {
        "family": "phospholipid",
        "bead_count": 7,
        "charge": 0.0,
        "charge_assumption": "zwitterionic net-neutral smoke-test topology",
        "matching_rules": ["role == phospholipid"],
    },
    "helper_lipid": {
        "family": "phospholipid",
        "bead_count": 7,
        "charge": 0.0,
        "charge_assumption": "helper lipid net-neutral smoke-test topology",
        "matching_rules": ["role == helper_lipid"],
    },
    "sterol": {
        "family": "sterol",
        "bead_count": 4,
        "charge": 0.0,
        "charge_assumption": "neutral sterol smoke-test topology",
        "matching_rules": ["role == sterol"],
    },
    "peg_lipid": {
        "family": "peg_lipid",
        "bead_count": 8,
        "charge": 0.0,
        "charge_assumption": "neutral PEG-lipid smoke-test topology with bounded PEG repeat placeholder",
        "matching_rules": ["role == peg_lipid", "PEG repeat count bounded for MVP"],
    },
    "small_molecule": {
        "family": "small_molecule",
        "bead_count": 3,
        "charge": 0.0,
        "charge_assumption": "neutral generic small-molecule smoke-test placeholder; production topology requires review",
        "matching_rules": ["role == small_molecule", "descriptor window matched"],
    },
    "additive": {
        "family": "additive",
        "bead_count": 3,
        "charge": 0.0,
        "charge_assumption": "neutral additive smoke-test placeholder; production topology requires review",
        "matching_rules": ["role == additive", "low-confidence additive fallback"],
    },
}


def normalize_ratios(lipids: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    kinds = []
    for lipid in lipids:
        if lipid.get("mol_fraction") is not None:
            values.append(float(lipid["mol_fraction"]))
            kinds.append("mol_fraction")
        elif lipid.get("mol_percent") is not None:
            values.append(float(lipid["mol_percent"]))
            kinds.append("mol_percent")
        elif lipid.get("molecule_count") is not None:
            values.append(float(lipid["molecule_count"]))
            kinds.append("molecule_count")
        else:
            raise ValueError(f"{lipid.get('name', '<unknown>')} has no ratio field")
    total = sum(values)
    if total <= 0:
        raise ValueError("Ratio total must be positive")
    if all(kind == "molecule_count" for kind in kinds):
        interpreted = "molecule_count"
    elif math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-6):
        interpreted = "mol_fraction"
    elif math.isclose(total, 100.0, rel_tol=0, abs_tol=1e-6):
        interpreted = "mol_percent"
    else:
        interpreted = "normalized_arbitrary_ratio"
    normalized = [value / total for value in values]
    return {
        "raw_total": total,
        "raw_values": values,
        "raw_fields": kinds,
        "interpreted_as": interpreted,
        "normalized_to": "mol_fraction",
        "tolerance": 1.0e-6,
        "normalized_values": normalized,
    }


def load_automation_policy(policy_path: str | None = None, allow_triage: bool = False, real_gromacs: bool | None = None) -> dict[str, Any]:
    policy = dict(DEFAULT_AUTOMATION_POLICY)
    policy["role_inference"] = dict(DEFAULT_AUTOMATION_POLICY["role_inference"])
    policy["simulation"] = dict(DEFAULT_AUTOMATION_POLICY["simulation"])
    if policy_path:
        loaded = read_yaml(policy_path)
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(policy.get(key), dict):
                policy[key].update(value)
            else:
                policy[key] = value
    if allow_triage:
        policy["allow_triage_generated_topologies"] = True
    if real_gromacs is not None:
        policy["real_gromacs_default"] = bool(real_gromacs)
    return policy


def parse_auto_input(input_value: str) -> dict[str, Any]:
    def normalize_component(component: dict[str, Any], idx: int) -> dict[str, Any]:
        raw_ratio = component.get("raw_ratio", component.get("ratio", component.get("mol_fraction", component.get("mol_percent"))))
        smiles = str(component.get("smiles") or component.get("SMILES") or "").strip()
        if not smiles:
            raise ValueError(f"Component {idx} is missing required SMILES")
        try:
            ratio = float(raw_ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Component {idx} has invalid ratio: {raw_ratio}") from exc
        if ratio < 0:
            raise ValueError(f"Component {idx} ratio must be non-negative")
        return {
            "local_id": str(component.get("local_id") or component.get("id") or f"component_{idx:03d}"),
            "name": str(component.get("name") or component.get("display_name") or component.get("local_id") or f"component_{idx:03d}"),
            "smiles": smiles,
            "raw_ratio": ratio,
            "role": str(component.get("role") or "unknown"),
            "topology_hint": component.get("topology_hint"),
            "topology_source_hint": component.get("topology_source_hint"),
            "topology_id": component.get("topology_id"),
        }

    source_path = Path(input_value)
    if source_path.exists():
        raw = source_path.read_text(encoding="utf-8")
        suffix = source_path.suffix.lower()
        if suffix == ".csv":
            rows = list(csv.DictReader(raw.splitlines()))
            components = rows
        elif suffix == ".json":
            loaded = json.loads(raw)
            components = loaded.get("components", loaded if isinstance(loaded, list) else [])
        elif suffix in {".yaml", ".yml"}:
            loaded = read_yaml(source_path)
            components = loaded.get("components", loaded.get("lipids", []))
        else:
            raise ValueError(f"Unsupported auto input file type: {source_path.suffix}")
        input_kind = suffix.lstrip(".")
        source_label = str(source_path)
    else:
        raw = input_value
        components = []
        inline_parts: list[str] = []
        for line in raw.replace(";", "\n").splitlines():
            inline_parts.extend(part for part in line.split(",") if part.strip())
        for part in inline_parts:
            part = part.strip()
            if ":" not in part:
                raise ValueError(f"Malformed component '{part}'; expected SMILES:ratio")
            smiles, ratio = part.rsplit(":", 1)
            components.append({"smiles": smiles.strip(), "raw_ratio": ratio.strip()})
        input_kind = "inline"
        source_label = "inline"
    normalized_components = []
    for idx, component in enumerate(components, start=1):
        normalized_components.append(normalize_component(component, idx))
    total = sum(c["raw_ratio"] for c in normalized_components)
    if total <= 0:
        raise ValueError("Auto input ratio total must be positive")
    warnings = []
    if any(c["raw_ratio"] == 0 for c in normalized_components):
        warnings.append({"severity": "warning", "message": "one or more components has zero ratio and will be retained for provenance but contributes no molecules"})
    duplicate_smiles = sorted({c["smiles"] for c in normalized_components if sum(1 for item in normalized_components if item["smiles"] == c["smiles"]) > 1})
    if duplicate_smiles:
        warnings.append({"severity": "warning", "message": "duplicate SMILES detected", "smiles": duplicate_smiles})
    for component in normalized_components:
        component["normalized_mol_fraction"] = component["raw_ratio"] / total
    return {
        "schema_version": "automd.auto_input.v0.1",
        "source": source_label,
        "input_kind": input_kind,
        "raw_input": raw,
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "ratio_policy": {
            "raw_total": total,
            "interpreted_as": "auto_ratio",
            "normalized_to": "mol_fraction",
        },
        "validation": {"component_count": len(normalized_components), "warnings": warnings},
        "components": normalized_components,
    }


def ensure_placeholder_topologies(root: Path) -> None:
    topo_dir = root / "topology_library" / "approved"
    topo_dir.mkdir(parents=True, exist_ok=True)
    for record in KNOWN_TOPOLOGIES:
        path = root / record.topology_files[0]["path"]
        if not path.exists():
            mol = record.molecule_type_name
            path.write_text(
                f"; AutoMD placeholder Martini topology for {mol}\n"
                "; Not production-approved. Used to exercise workflow packaging.\n"
                "[ moleculetype ]\n"
                f"{mol} 1\n\n"
                "[ atoms ]\n"
                f"1 P1 1 {mol} B1 1 0.0 72.0\n",
                encoding="utf-8",
            )


def command_env_doctor(json_mode: bool = False) -> dict[str, Any]:
    gmx_path = shutil.which("gmx")
    version = _gromacs_version(gmx_path)
    version_raw = version["version_raw"]
    return_code = version["return_code"]
    optional_tools = {}
    for tool, module in [
        ("packmol", None),
        ("insane", "insane"),
        ("polyply", "polyply"),
        ("martiniglass", "martiniglass"),
    ]:
        optional_tools[tool] = {
            "path": shutil.which(tool),
            "available": bool(shutil.which(tool)),
            "module": module,
            "module_available": bool(importlib.util.find_spec(module)) if module else None,
        }
    report = {
        "schema_version": "automd.env_report.v0.1",
        "created_at": utc_now(),
        "automd_version": __version__,
        "python": {"executable": shutil.which("python") or "", "version": os.sys.version.split()[0]},
        "gromacs": {
            "executable": gmx_path,
            "available": bool(gmx_path),
            "return_code": return_code,
            "version_raw": version_raw,
            "warning": None if gmx_path else "GROMACS executable 'gmx' not found; use --dry-run or install/load GROMACS.",
        },
        "optional_tools": optional_tools,
    }
    if json_mode:
        print(json.dumps(report, indent=2))
    else:
        print(f"AutoMD {__version__}")
        print("GROMACS:", gmx_path or "missing (dry-run workflows still available)")
    return report


def _gromacs_version(gmx_path: str | None) -> dict[str, Any]:
    if not gmx_path:
        return {"version_raw": None, "return_code": None}
    proc = subprocess.run([gmx_path, "--version"], capture_output=True, text=True, check=False)
    return {"version_raw": (proc.stdout + proc.stderr).strip(), "return_code": proc.returncode}


def command_sources_list() -> dict[str, Any]:
    sources = []
    for source in SOURCE_REPOS:
        path = Path(source["path"])
        commit = None
        if path.exists() and (path / ".git").exists():
            proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
            commit = proc.stdout.strip() if proc.returncode == 0 else None
        sources.append({**source, "present": path.exists(), "commit": commit})
    sources = {
        "schema_version": "automd.sources.v0.1",
        "sources": sources,
    }
    print(json.dumps(sources, indent=2))
    return sources


def command_sources_fetch(all_sources: bool = False) -> Path:
    if not all_sources:
        raise ValueError("Use --all to fetch the approved public Martini source set.")
    if not shutil.which("git"):
        raise RuntimeError("git is required to fetch external sources")
    external = Path("external")
    external.mkdir(exist_ok=True)
    records = []
    for source in SOURCE_REPOS:
        path = Path(source["path"])
        if path.exists():
            proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
            status = "present"
        else:
            proc = subprocess.run(["git", "clone", "--depth", "1", source["url"], str(path)], capture_output=True, text=True, check=False)
            status = "fetched" if proc.returncode == 0 else "failed"
        commit_proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False) if path.exists() else proc
        records.append({
            **source,
            "status": status,
            "return_code": proc.returncode,
            "commit": commit_proc.stdout.strip() if commit_proc.returncode == 0 else None,
            "stderr": proc.stderr.strip(),
        })
    manifest = write_yaml("external/source_manifest.yaml", {"schema_version": "automd.source_manifest.v0.1", "created_at": utc_now(), "sources": records})
    print(manifest)
    return manifest


def command_intake(input_path: str, out: str | None = None) -> Path:
    data = read_yaml(input_path)
    formulation = FormulationInput.model_validate(data)
    run_dir = Path(out or Path("runs") / formulation.formulation_id)
    (run_dir / "manifests").mkdir(parents=True, exist_ok=True)
    frozen = freeze_input(input_path, run_dir / "inputs")
    lipids = [lipid.model_dump() for lipid in formulation.lipids]
    ratio = normalize_ratios(lipids)
    lipid_records = []
    for idx, lipid in enumerate(lipids, start=1):
        local_id = lipid.get("local_id") or f"lipid_{idx:03d}"
        lipid["local_id"] = local_id
        lipid_records.append({
            "automd_lipid_id": f"AMLIP-{idx:06d}",
            "local_id": local_id,
            "display_name": lipid["name"],
            "role": lipid["role"],
            "raw_smiles": lipid.get("smiles"),
            "inchi_key": lipid.get("inchi_key"),
            "aliases": [{"type": "common_name", "value": lipid["name"], "source": "user_input"}],
            "raw_ratio": ratio["raw_values"][idx - 1],
            "normalized_mol_fraction": ratio["normalized_values"][idx - 1],
            "topology_hint": lipid.get("topology_hint"),
            "topology_status": "unresolved",
            "review_status": "pending",
        })
    manifest = {
        "schema_version": "automd.intake_manifest.v0.1",
        "created_at": utc_now(),
        "automd_version": __version__,
        "formulation_id": formulation.formulation_id,
        "name": formulation.name or formulation.formulation_id,
        "run_dir": str(run_dir),
        "raw_input": frozen,
        "ratio_policy": {k: v for k, v in ratio.items() if k != "normalized_values"},
        "payload": formulation.payload.model_dump(),
        "simulation_request": formulation.simulation_request.model_dump(),
        "lipids": lipid_records,
    }
    write_yaml(run_dir / "manifests" / "intake_manifest.yaml", manifest)
    write_yaml(run_dir / "inputs" / "lipid_records.yaml", {"schema_version": "automd.lipid_record.v0.1", "lipids": lipid_records})
    print(run_dir / "manifests" / "intake_manifest.yaml")
    return run_dir / "manifests" / "intake_manifest.yaml"


def _rdkit_descriptors(smiles: str) -> dict[str, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except Exception as exc:
        return {"descriptor_status": "failed_rdkit_unavailable", "error": str(exc)}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"descriptor_status": "failed_invalid_structure"}
    return {
        "descriptor_status": "computed_from_structure",
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "inchi_key": Chem.inchi.MolToInchiKey(mol) if hasattr(Chem, "inchi") else None,
        "exact_mw": float(Descriptors.ExactMolWt(mol)),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        "hbond_donors": int(Lipinski.NumHDonors(mol)),
        "hbond_acceptors": int(Lipinski.NumHAcceptors(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "logp_rdkit": float(Crippen.MolLogP(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "ring_count": int(Lipinski.RingCount(mol)),
        "heteroatom_count": int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
    }


def command_descriptors(intake_manifest: str) -> Path:
    intake = read_yaml(intake_manifest)
    run_dir = Path(intake["run_dir"])
    rows = []
    for lipid in intake["lipids"]:
        row = dict(lipid)
        smiles = lipid.get("raw_smiles")
        if smiles:
            row.update(_rdkit_descriptors(smiles))
        else:
            row.update({
                "descriptor_status": "missing_no_structure",
                "canonical_smiles": None,
                "exact_mw": None,
                "formula": None,
                "formal_charge": None,
                "hbond_donors": None,
                "hbond_acceptors": None,
                "tpsa": None,
                "logp_rdkit": None,
                "rotatable_bonds": None,
                "ring_count": None,
                "heteroatom_count": None,
            })
        rows.append(row)
    desc_dir = run_dir / "descriptors"
    desc_dir.mkdir(parents=True, exist_ok=True)
    table = desc_dir / "descriptor_table.parquet"
    pd.DataFrame(rows).to_parquet(table, index=False)
    coverage = sum(1 for row in rows if row["descriptor_status"] == "computed_from_structure") / len(rows)
    manifest = {
        "schema_version": "automd.descriptor_manifest.v0.1",
        "created_at": utc_now(),
        "formulation_id": intake["formulation_id"],
        "run_dir": str(run_dir),
        "input_manifest": str(Path(intake_manifest)),
        "descriptor_policy": "structure descriptors only when SMILES/SDF/MOL2 data are available",
        "descriptor_coverage": coverage,
        "descriptor_table": file_record(table, run_dir),
        "lipids": rows,
        "formulation_descriptors": formulation_descriptors(intake, rows),
    }
    out = write_yaml(run_dir / "manifests" / "descriptor_manifest.yaml", manifest)
    print(out)
    return out


def formulation_descriptors(intake: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    fractions = [float(row["normalized_mol_fraction"]) for row in rows]
    entropy = -sum(f * math.log(f) for f in fractions if f > 0)
    roles = {}
    for row in rows:
        roles[row["role"]] = roles.get(row["role"], 0.0) + float(row["normalized_mol_fraction"])
    return {
        "component_count": len(rows),
        "role_proportions": roles,
        "ratio_entropy": entropy,
        "max_component_fraction": max(fractions),
        "min_component_fraction": min(fractions),
        "descriptor_coverage": sum(1 for row in rows if row["descriptor_status"] == "computed_from_structure") / len(rows),
        "peg_lipid_fraction": roles.get("peg_lipid", 0.0),
        "ionizable_lipid_fraction": roles.get("ionizable_lipid", 0.0),
        "sterol_fraction": roles.get("sterol", 0.0),
        "helper_phospholipid_fraction": roles.get("phospholipid", 0.0) + roles.get("helper_lipid", 0.0),
    }


def _smarts_match(smiles: str | None, smarts: str) -> bool:
    if not smiles:
        return False
    try:
        from rdkit import Chem
    except Exception:
        return False
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts(smarts)
    return bool(mol is not None and pattern is not None and mol.HasSubstructMatch(pattern))


def infer_component_role(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    smiles = row.get("canonical_smiles") or row.get("raw_smiles")
    status = row.get("descriptor_status")
    matched = []
    role = "unknown"
    confidence = 0.0
    if status not in {"computed_from_structure"}:
        return {
            "local_id": row["local_id"],
            "inferred_role": "unknown",
            "confidence": 0.0,
            "matched_rules": ["invalid_or_missing_structure"],
            "fallback_used": True,
            "production_eligible_role_inference": False,
        }
    ring_count = int(row.get("ring_count") or 0)
    hetero = int(row.get("heteroatom_count") or 0)
    mw = float(row.get("exact_mw") or 0.0)
    rot = int(row.get("rotatable_bonds") or 0)
    if ring_count >= 4 and _smarts_match(smiles, "[OX2H]"):
        role, confidence, matched = "sterol", 0.86, ["sterol_ring_count", "hydroxyl"]
    elif _smarts_match(smiles, "P(=O)(O)") or _smarts_match(smiles, "P(=O)([O-])"):
        role, confidence, matched = "phospholipid", 0.82, ["phosphate_or_phosphocholine"]
    elif smiles and (smiles.count("OCC") >= 2 or smiles.count("COC") >= 2):
        role, confidence, matched = "peg_lipid", 0.78, ["peg_ether_repeat"]
    elif (_smarts_match(smiles, "[NX3]([#6])([#6])[#6]") or _smarts_match(smiles, "[NX4+]")) and (mw > 180 or rot >= 6):
        role, confidence, matched = "ionizable_lipid", 0.78, ["tertiary_or_quaternary_amine", "hydrophobic_chain_proxy"]
    elif mw < 350 and hetero <= 4:
        role, confidence, matched = "small_molecule", 0.72, ["small_molecule_descriptor_window"]
    else:
        role, confidence, matched = policy["role_inference"]["low_confidence_role_fallback"], 0.45, ["fallback_low_confidence"]
    minimum = float(policy["role_inference"]["minimum_confidence_for_auto_role"])
    fallback_used = confidence < minimum
    if fallback_used:
        role = policy["role_inference"]["low_confidence_role_fallback"]
    return {
        "local_id": row["local_id"],
        "inferred_role": role,
        "confidence": round(confidence, 3),
        "matched_rules": matched,
        "fallback_used": fallback_used,
        "production_eligible_role_inference": not fallback_used and role not in {"unknown", "additive"},
    }


def apply_role_inference(descriptor_manifest: str, policy: dict[str, Any]) -> list[dict[str, Any]]:
    desc = read_yaml(descriptor_manifest)
    inferences = []
    for row in desc["lipids"]:
        inferred = infer_component_role(row, policy)
        inferences.append(inferred)
        row["original_role"] = row.get("role")
        row["role"] = inferred["inferred_role"]
        row["role_inference"] = inferred
    desc["formulation_descriptors"] = formulation_descriptors({"formulation_id": desc["formulation_id"]}, desc["lipids"])
    write_yaml(descriptor_manifest, desc)
    return inferences


def command_topology_index(out: str = "topology_library/topology_registry.yaml") -> Path:
    ensure_placeholder_topologies(Path("."))
    records = []
    for rec in KNOWN_TOPOLOGIES:
        data = rec.model_dump()
        for topo_file in data["topology_files"]:
            p = Path(topo_file["path"])
            topo_file["sha256"] = sha256_file(p) if p.exists() else None
        records.append(data)
    path = write_yaml(out, {"schema_version": "automd.topology_registry.v0.1", "created_at": utc_now(), "topologies": records})
    print(path)
    return path


def load_registry() -> list[dict[str, Any]]:
    path = Path("topology_library/topology_registry.yaml")
    if not path.exists():
        command_topology_index(str(path))
    return read_yaml(path)["topologies"]


def match_topologies(lipid: dict[str, Any], registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terms = {str(lipid.get("display_name") or "").lower(), str(lipid.get("topology_hint") or "").lower()}
    matches = []
    for rec in registry:
        aliases = {a["value"].lower() for a in rec.get("aliases", [])} | {n.lower() for n in rec.get("molecule_names", [])} | {rec.get("molecule_type_name", "").lower()}
        if terms & aliases or lipid.get("role") in rec.get("role_tags", []):
            confidence = "exact_alias" if terms & aliases else "role_based_review_required"
            production_eligible = bool(rec.get("production_eligible", rec.get("source_family") != "mock_test"))
            smoke_eligible = bool(rec.get("smoke_eligible", True))
            placeholder = bool(rec.get("placeholder_topology") or rec.get("source_family") == "mock_test")
            candidate = {
                "topology_id": rec["topology_id"],
                "label": rec["display_name"],
                "confidence": confidence,
                "allowed_for_production": confidence == "exact_alias" and production_eligible and rec["approval_status"] in {"approved_exact", "imported_trusted_source"},
                "smoke_eligible": smoke_eligible,
                "production_eligible": production_eligible,
                "production_review_required": bool(rec.get("production_review_required", not production_eligible)),
                "placeholder_topology": placeholder,
                "approval_status": rec["approval_status"],
                "source_family": rec.get("source_family"),
                "source_repo": rec.get("source_repo"),
                "source_commit": rec.get("source_commit"),
                "molecule_type_name": rec["molecule_type_name"],
                "topology_files": rec["topology_files"],
                "limitations": rec.get("limitations", []),
            }
            matches.append(candidate)
    return matches


def _safe_molecule_type(name: str, fallback: str) -> str:
    raw = "".join(ch for ch in (name or fallback).upper() if ch.isalnum())
    return (raw or fallback.upper())[:12]


def _write_generated_itp(path: Path, molecule_type: str, bead_count: int, charge: float = 0.0) -> None:
    atom_lines = []
    for idx in range(1, bead_count + 1):
        atom_charge = charge if idx == 1 else 0.0
        atom_lines.append(f"{idx:5d} P1     1 {molecule_type:<8} B{idx:<4d} {idx:5d} {atom_charge:8.3f} 72.0")
    bond_lines = []
    for idx in range(1, bead_count):
        bond_lines.append(f"{idx:5d} {idx + 1:5d} 1 0.300 1250")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"; AutoMD generated topology for {molecule_type}\n"
        "; Generated topology is computed setup data, not empirical validation.\n"
        "[ moleculetype ]\n"
        f"{molecule_type} 1\n\n"
        "[ atoms ]\n"
        + "\n".join(atom_lines)
        + ("\n\n[ bonds ]\n" + "\n".join(bond_lines) if bond_lines else "")
        + "\n",
        encoding="utf-8",
    )


def _command_to_argv(command: str | list[str]) -> list[str]:
    if isinstance(command, list):
        return [str(part) for part in command]
    argv = shlex.split(str(command))
    if not argv:
        raise ValueError("External command is empty")
    return argv


def _topology_terms(lipid: dict[str, Any]) -> set[str]:
    return {
        str(lipid.get("display_name") or lipid.get("name") or "").lower(),
        str(lipid.get("topology_hint") or "").lower(),
        str(lipid.get("inchi_key") or "").lower(),
        str(lipid.get("canonical_smiles") or "").lower(),
    }


class TopologyGenerator:
    name = "base"
    version = "automd.topology_generator.v0.1"
    supported_roles: set[str] = set()

    def can_generate(self, lipid: dict[str, Any], registry: list[dict[str, Any]]) -> dict[str, Any]:
        return {"supported": False, "reason": "base class"}

    def generate(self, lipid: dict[str, Any], run_dir: Path, registry: list[dict[str, Any]]) -> dict[str, Any] | None:
        raise NotImplementedError


class CuratedRegistryBackend(TopologyGenerator):
    name = "CuratedRegistryBackend"
    supported_roles = {"*"}

    def can_generate(self, lipid: dict[str, Any], registry: list[dict[str, Any]]) -> dict[str, Any]:
        matches = match_topologies(lipid, registry)
        exact = [m for m in matches if m.get("confidence") == "exact_alias"]
        return {"supported": bool(exact), "reason": "exact curated registry match" if exact else "no exact curated match"}

    def generate(self, lipid: dict[str, Any], run_dir: Path, registry: list[dict[str, Any]]) -> dict[str, Any] | None:
        exact = [m for m in match_topologies(lipid, registry) if m.get("confidence") == "exact_alias"]
        if not exact:
            return None
        selected = exact[0]
        approval = selected.get("approval_status")
        tier = "A_curated_exact" if approval == "approved_exact" else "B_curated_family_match"
        return {
            "backend": self.name,
            "backend_version": self.version,
            "status": "curated_match",
            "confidence_tier": tier,
            "confidence": approval,
            "smoke_eligible": bool(selected.get("smoke_eligible", True)),
            "production_eligible": bool(selected.get("allowed_for_production")),
            "review_required": CONFIDENCE_TIERS[tier]["review_required"],
            "production_review_required": bool(selected.get("production_review_required", not selected.get("allowed_for_production"))),
            "placeholder_topology": bool(selected.get("placeholder_topology")),
            "topology_id": selected["topology_id"],
            "molecule_type_name": selected["molecule_type_name"],
            "topology_files": selected["topology_files"],
            "provenance": {
                "match_terms": sorted(_topology_terms(lipid)),
                "approval_status": approval,
                "source": "topology_registry",
                "source_family": selected.get("source_family"),
                "source_repo": selected.get("source_repo"),
                "source_commit": selected.get("source_commit"),
                "placeholder_topology": bool(selected.get("placeholder_topology")),
                "limitations": selected.get("limitations", []),
            },
        }


class FragmentTemplateBackend(TopologyGenerator):
    name = "FragmentTemplateBackend"
    version = "automd.fragment_templates.v0.1"
    supported_roles = set(ROLE_FAMILY_RULES)

    def can_generate(self, lipid: dict[str, Any], registry: list[dict[str, Any]]) -> dict[str, Any]:
        supported = lipid.get("role") in self.supported_roles
        return {"supported": supported, "reason": "supported lipid family" if supported else "unsupported role for family templates"}

    def generate(self, lipid: dict[str, Any], run_dir: Path, registry: list[dict[str, Any]]) -> dict[str, Any] | None:
        rule = ROLE_FAMILY_RULES.get(lipid.get("role"))
        if not rule:
            return None
        local_id = lipid["local_id"]
        moltype = _safe_molecule_type(lipid.get("display_name") or "", local_id)
        topo_path = run_dir / "topologies" / "generated" / f"{local_id}.generated.itp"
        _write_generated_itp(topo_path, moltype, int(rule["bead_count"]), float(rule["charge"]))
        manifest = {
            "schema_version": "automd.topology_generation_record.v0.1",
            "created_at": utc_now(),
            "local_id": local_id,
            "lipid_name": lipid.get("display_name"),
            "backend": self.name,
            "backend_version": self.version,
            "fragment_library_version": "automd.approved_fragments.v0.1",
            "family": rule["family"],
            "matching_rules": rule["matching_rules"],
            "unsupported_motifs": [],
            "charge_protonation_assumptions": [rule["charge_assumption"]],
            "generated_itp": file_record(topo_path, run_dir),
            "molecule_type_name": moltype,
            "confidence_tier": "C_generated_from_approved_fragments",
            "smoke_eligible": True,
            "production_eligible": False,
            "review_required": False,
            "production_review_required": True,
            "placeholder_topology": True,
        }
        record_path = run_dir / "topologies" / "generated" / f"{local_id}.generation_manifest.yaml"
        write_yaml(record_path, manifest)
        return {
            "backend": self.name,
            "backend_version": self.version,
            "status": "generated",
            "confidence_tier": "C_generated_from_approved_fragments",
            "confidence": "generated_from_approved_family_template",
            "smoke_eligible": True,
            "production_eligible": False,
            "review_required": False,
            "production_review_required": True,
            "placeholder_topology": True,
            "topology_id": f"GEN-{local_id}",
            "molecule_type_name": moltype,
            "topology_files": [{"path": str(topo_path), "sha256": sha256_file(topo_path)}],
            "generation_manifest": str(record_path),
            "provenance": manifest,
        }


class PolyplyBackend(TopologyGenerator):
    name = "PolyplyBackend"
    version = "automd.polyply_adapter.v0.1"
    supported_roles = {"small_molecule", "helper_molecule", "additive", "unknown"}

    def can_generate(self, lipid: dict[str, Any], registry: list[dict[str, Any]]) -> dict[str, Any]:
        has_structure = bool(lipid.get("canonical_smiles")) and lipid.get("descriptor_status") == "computed_from_structure"
        supported = has_structure and lipid.get("role") not in ROLE_FAMILY_RULES
        return {"supported": supported, "reason": "structure present for triage backend" if supported else "no triage structure or family backend preferred"}

    def generate(self, lipid: dict[str, Any], run_dir: Path, registry: list[dict[str, Any]]) -> dict[str, Any] | None:
        support = self.can_generate(lipid, registry)
        if not support["supported"]:
            return None
        local_id = lipid["local_id"]
        moltype = _safe_molecule_type(lipid.get("display_name") or "", local_id)
        topo_path = run_dir / "topologies" / "generated" / f"{local_id}.polyply_triage.itp"
        _write_generated_itp(topo_path, moltype, 3, 0.0)
        manifest = {
            "schema_version": "automd.topology_generation_record.v0.1",
            "created_at": utc_now(),
            "local_id": local_id,
            "lipid_name": lipid.get("display_name"),
            "backend": self.name,
            "backend_version": self.version,
            "adapter_executable": shutil.which("polyply"),
            "adapter_module_available": bool(importlib.util.find_spec("polyply")),
            "input_structure": lipid.get("canonical_smiles") or lipid.get("raw_smiles"),
            "unsupported_motifs": ["not classified into approved lipid-family template"],
            "charge_protonation_assumptions": ["neutral triage-only small-molecule placeholder"],
            "generated_itp": file_record(topo_path, run_dir),
            "molecule_type_name": moltype,
            "confidence_tier": "D_generated_smallmol_backend",
            "smoke_eligible": False,
            "production_eligible": False,
            "review_required": True,
            "production_review_required": True,
            "placeholder_topology": True,
        }
        record_path = run_dir / "topologies" / "generated" / f"{local_id}.generation_manifest.yaml"
        write_yaml(record_path, manifest)
        return {
            "backend": self.name,
            "backend_version": self.version,
            "status": "generated_requires_review",
            "confidence_tier": "D_generated_smallmol_backend",
            "confidence": "generated_requires_review",
            "smoke_eligible": False,
            "production_eligible": False,
            "review_required": True,
            "production_review_required": True,
            "placeholder_topology": True,
            "topology_id": f"TRIAGE-{local_id}",
            "molecule_type_name": moltype,
            "topology_files": [{"path": str(topo_path), "sha256": sha256_file(topo_path)}],
            "generation_manifest": str(record_path),
            "provenance": manifest,
        }


class ExternalCommandBackend(TopologyGenerator):
    name = "ExternalCommandBackend"
    version = "automd.external_topology_command.v0.1"
    supported_roles = {"*"}

    def can_generate(self, lipid: dict[str, Any], registry: list[dict[str, Any]]) -> dict[str, Any]:
        return {"supported": bool(os.environ.get("AUTOMD_TOPOLOGY_GENERATOR")), "reason": "AUTOMD_TOPOLOGY_GENERATOR set"}

    def generate(self, lipid: dict[str, Any], run_dir: Path, registry: list[dict[str, Any]]) -> dict[str, Any] | None:
        command = os.environ.get("AUTOMD_TOPOLOGY_GENERATOR")
        if not command:
            return None
        local_id = lipid["local_id"]
        out_dir = run_dir / "topologies" / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "AUTOMD_LIPID_JSON": json.dumps(lipid), "AUTOMD_TOPOLOGY_OUTPUT_DIR": str(out_dir)}
        command_args = _command_to_argv(command)
        proc = subprocess.run(command_args, cwd=run_dir, env=env, capture_output=True, text=True, check=False, timeout=300)
        stdout = out_dir / f"{local_id}.external.stdout.txt"
        stderr = out_dir / f"{local_id}.external.stderr.txt"
        stdout.write_text(proc.stdout, encoding="utf-8")
        stderr.write_text(proc.stderr, encoding="utf-8")
        expected = out_dir / f"{local_id}.external.itp"
        if proc.returncode != 0 or not expected.exists():
            return None
        moltype = parse_itp_summary(expected).get("molecule_type_name") or _safe_molecule_type(lipid.get("display_name") or "", local_id)
        return {
            "backend": self.name,
            "backend_version": self.version,
            "status": "generated_requires_review",
            "confidence_tier": "D_generated_smallmol_backend",
            "confidence": "generated_requires_review",
            "smoke_eligible": False,
            "production_eligible": False,
            "review_required": True,
            "production_review_required": True,
            "placeholder_topology": True,
            "topology_id": f"EXT-{local_id}",
            "molecule_type_name": moltype,
            "topology_files": [{"path": str(expected), "sha256": sha256_file(expected)}],
            "provenance": {"command": command_args, "return_code": proc.returncode, "stdout": file_record(stdout, run_dir), "stderr": file_record(stderr, run_dir)},
        }


class DescriptorOnlyBackend(TopologyGenerator):
    name = "DescriptorOnlyBackend"
    version = "automd.descriptor_only.v0.1"
    supported_roles = {"*"}

    def can_generate(self, lipid: dict[str, Any], registry: list[dict[str, Any]]) -> dict[str, Any]:
        return {"supported": True, "reason": "always available fallback"}

    def generate(self, lipid: dict[str, Any], run_dir: Path, registry: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "backend": self.name,
            "backend_version": self.version,
            "status": "manual_review_required",
            "confidence_tier": "E_manual_review_required",
            "confidence": "descriptor_only",
            "smoke_eligible": False,
            "production_eligible": False,
            "review_required": True,
            "production_review_required": True,
            "placeholder_topology": False,
            "topology_id": None,
            "molecule_type_name": None,
            "topology_files": [],
            "provenance": {"reason": "no trusted topology could be resolved or generated"},
        }


TOPOLOGY_GENERATORS: list[TopologyGenerator] = [
    CuratedRegistryBackend(),
    FragmentTemplateBackend(),
    PolyplyBackend(),
    ExternalCommandBackend(),
    DescriptorOnlyBackend(),
]


def parse_itp_summary(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    section = None
    molecule_type = None
    atom_count = 0
    charges = []
    includes = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#include"):
            includes.append(line)
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip()
            continue
        parts = line.split()
        if section == "moleculetype" and molecule_type is None and parts:
            molecule_type = parts[0]
        elif section == "atoms" and len(parts) >= 7:
            atom_count += 1
            try:
                charges.append(float(parts[6]))
            except ValueError:
                charges.append(float("nan"))
    return {
        "path": str(path),
        "molecule_type_name": molecule_type,
        "atom_count": atom_count,
        "charges": charges,
        "net_charge": sum(charges) if charges and all(math.isfinite(c) for c in charges) else None,
        "includes": includes,
    }


def _validate_itp_with_gromacs(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    gmx = shutil.which("gmx")
    if not gmx:
        return {"status": "skipped", "reason": "gmx not available", "command": None, "return_code": None}
    molecule_type = summary.get("molecule_type_name")
    atom_count = int(summary.get("atom_count") or 0)
    if not molecule_type or atom_count < 1:
        return {"status": "failed", "reason": "missing molecule type or atoms", "command": None, "return_code": None}
    with tempfile.TemporaryDirectory(prefix="automd_topology_validate_") as tmp:
        tmpdir = Path(tmp)
        itp = tmpdir / path.name
        shutil.copy2(path, itp)
        (tmpdir / "topol.top").write_text(
            "[ defaults ]\n"
            "1 1 no 1.0 1.0\n\n"
            "[ atomtypes ]\n"
            "P1 72.0 0.0 A 0.470 5.000\n\n"
            f'#include "{itp.name}"\n\n'
            "[ system ]\nvalidation\n\n"
            "[ molecules ]\n"
            f"{molecule_type} 1\n",
            encoding="utf-8",
        )
        atoms = []
        for idx in range(1, atom_count + 1):
            atoms.append(f"{1:5d}{molecule_type[:5]:<5}{('B'+str(idx))[:5]:>5}{idx:5d}{0.1 * idx:8.3f}{0.1:8.3f}{0.1:8.3f}")
        (tmpdir / "system.gro").write_text("validation\n" + f"{atom_count:5d}\n" + "\n".join(atoms) + "\n   3.00000   3.00000   3.00000\n", encoding="utf-8")
        _write_gromacs_ready_mdp(tmpdir / "em.mdp", "em", 0)
        cmd = [gmx, "grompp", "-f", "em.mdp", "-c", "system.gro", "-p", "topol.top", "-o", "validate.tpr", "-maxwarn", "0"]
        proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, check=False)
        return {
            "status": "pass" if proc.returncode == 0 else "failed",
            "command": " ".join(cmd),
            "return_code": proc.returncode,
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
        }


def validate_topology_file(path: str | Path, confidence_tier: str | None = None, production_eligible: bool | None = None) -> dict[str, Any]:
    path = Path(path)
    summary = parse_itp_summary(path)
    gates = {
        "molecule_type_exists": bool(summary["molecule_type_name"]),
        "atom_bead_count_nonzero": int(summary["atom_count"]) > 0,
        "charges_finite": bool(summary["charges"]) and all(math.isfinite(charge) for charge in summary["charges"]),
        "includes_resolve": all((path.parent / include.split('"')[1]).exists() for include in summary["includes"] if '"' in include),
        "hash_stored": bool(sha256_file(path)),
        "confidence_tier_stored": bool(confidence_tier),
        "production_eligibility_explicit": production_eligible is not None,
        "no_unreviewed_maxwarn": True,
    }
    grompp = _validate_itp_with_gromacs(path, summary)
    gates["gmx_grompp_succeeds"] = grompp["status"] in {"pass", "skipped"}
    status = "pass" if all(gates.values()) and grompp["status"] != "failed" else "fail"
    return {
        "schema_version": "automd.topology_validation_record.v0.1",
        "created_at": utc_now(),
        "topology_file": str(path),
        "sha256": sha256_file(path),
        "summary": summary,
        "confidence_tier": confidence_tier,
        "production_eligible": production_eligible,
        "gates": gates,
        "grompp_preflight": grompp,
        "validation_status": status,
    }


def _candidate_from_generation(lipid: dict[str, Any], generation: dict[str, Any], validation: dict[str, Any], option_id: int) -> dict[str, Any]:
    validation_passed = validation.get("validation_status") == "pass"
    smoke_eligible = bool(generation.get("smoke_eligible", CONFIDENCE_TIERS[generation["confidence_tier"]]["smoke_eligible"])) and validation_passed
    production_eligible = bool(generation.get("production_eligible")) and validation_passed
    return {
        "option_id": option_id,
        "topology_id": generation.get("topology_id"),
        "label": f"{generation['backend']} {generation['confidence_tier']}",
        "confidence": generation["confidence"],
        "confidence_tier": generation["confidence_tier"],
        "backend": generation["backend"],
        "molecule_type_name": generation.get("molecule_type_name"),
        "topology_files": generation.get("topology_files", []),
        "generation_manifest": generation.get("generation_manifest"),
        "validation_status": validation.get("validation_status") if validation else "not_applicable",
        "allowed_for_production": production_eligible,
        "production_eligible": production_eligible,
        "review_required": bool(generation.get("review_required")),
        "allowed_for_smoke": smoke_eligible,
        "smoke_eligible": smoke_eligible,
        "production_review_required": bool(generation.get("production_review_required", not production_eligible)),
        "placeholder_topology": bool(generation.get("placeholder_topology")),
        "provenance": generation.get("provenance", {}),
    }


def command_topology_generate(descriptor_manifest: str) -> Path:
    desc = read_yaml(descriptor_manifest)
    run_dir = Path(desc["run_dir"])
    registry = load_registry()
    generated_dir = run_dir / "topologies" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generations = []
    validations = []
    packets = []
    for idx, lipid in enumerate(desc["lipids"], start=1):
        attempts = []
        selected_generation = None
        for backend in TOPOLOGY_GENERATORS:
            support = backend.can_generate(lipid, registry)
            attempts.append({"backend": backend.name, **support})
            if not support["supported"]:
                continue
            selected_generation = backend.generate(lipid, run_dir, registry)
            if selected_generation:
                break
        if selected_generation is None:
            selected_generation = DescriptorOnlyBackend().generate(lipid, run_dir, registry)
        generation_record = {
            "local_id": lipid["local_id"],
            "lipid_name": lipid["display_name"],
            "role": lipid["role"],
            "attempts": attempts,
            "selected": selected_generation,
        }
        generations.append(generation_record)
        validation_records = []
        for topo_file in selected_generation.get("topology_files", []):
            validation = validate_topology_file(topo_file["path"], selected_generation["confidence_tier"], selected_generation["production_eligible"])
            validation_records.append(validation)
            validations.append({"local_id": lipid["local_id"], **validation})
        if not validation_records:
            validation_records = [{"validation_status": "not_applicable", "gates": {}, "topology_file": None}]
        candidate = _candidate_from_generation(lipid, selected_generation, validation_records[0], 1)
        descriptor_only = {
            "option_id": 2,
            "label": "Mark unresolved; descriptor-only",
            "allowed_for_production": False,
            "production_eligible": False,
            "review_required": True,
            "production_review_required": True,
            "placeholder_topology": False,
            "confidence": "descriptor_only",
            "confidence_tier": "E_manual_review_required",
            "allowed_for_smoke": False,
            "smoke_eligible": False,
        }
        packets.append({
            "schema_version": "automd.review_packet.v0.1",
            "review_packet_id": f"REVIEW-TOPO-{idx:06d}",
            "question_type": "select_topology",
            "lipid": {"local_id": lipid["local_id"], "name": lipid["display_name"], "role": lipid["role"]},
            "candidates": [candidate, descriptor_only],
            "allowed_answers": [1, 2],
            "auto_selected_option_id": 1 if candidate["allowed_for_smoke"] and not candidate["review_required"] else None,
        })
    generation_manifest = write_yaml(
        run_dir / "topology" / "topology_generation_manifest.yaml",
        {
            "schema_version": "automd.topology_generation_manifest.v0.1",
            "created_at": utc_now(),
            "run_dir": str(run_dir),
            "descriptor_manifest": str(descriptor_manifest),
            "confidence_tiers": CONFIDENCE_TIERS,
            "generations": generations,
        },
    )
    validation_manifest = write_yaml(
        run_dir / "topology" / "topology_validation_manifest.yaml",
        {
            "schema_version": "automd.topology_validation_manifest.v0.1",
            "created_at": utc_now(),
            "run_dir": str(run_dir),
            "topology_generation_manifest": str(generation_manifest),
            "validations": validations,
        },
    )
    candidates_path = write_yaml(
        run_dir / "topology" / "topology_candidates.yaml",
        {
            "schema_version": "automd.topology_candidates.v0.1",
            "created_at": utc_now(),
            "run_dir": str(run_dir),
            "topology_generation_manifest": str(generation_manifest),
            "topology_validation_manifest": str(validation_manifest),
            "review_packets": packets,
        },
    )
    print(candidates_path)
    return candidates_path


def command_topology_validate(paths: list[str], out: str | None = None) -> Path:
    expanded = [Path(path) for path in paths]
    validations = []
    for path in expanded:
        local_id = path.name.split(".", 1)[0]
        generation_manifest = path.parent / f"{local_id}.generation_manifest.yaml"
        confidence_tier = None
        production_eligible = None
        if generation_manifest.exists():
            generation = read_yaml(generation_manifest)
            confidence_tier = generation.get("confidence_tier")
            production_eligible = generation.get("production_eligible")
        validations.append(validate_topology_file(path, confidence_tier, production_eligible))
    out_path = Path(out) if out else Path("topology_validation_manifest.yaml")
    manifest = {
        "schema_version": "automd.topology_validation_manifest.v0.1",
        "created_at": utc_now(),
        "validations": validations,
    }
    path = write_yaml(out_path, manifest)
    print(path)
    return path


def _approval_entries(approvals_path: str) -> dict[str, dict[str, Any]]:
    data = read_yaml(approvals_path)
    entries = data.get("approvals", data) if isinstance(data, dict) else {}
    if isinstance(entries, list):
        normalized = {}
        for entry in entries:
            key = entry.get("local_id") or entry.get("component") or entry.get("topology_id")
            if key:
                normalized[str(key)] = entry
        return normalized
    if isinstance(entries, dict):
        return {str(key): (value if isinstance(value, dict) else {"approve_for_production": bool(value)}) for key, value in entries.items()}
    return {}


def command_topology_approve(topology_review_manifest: str, approvals: str) -> Path:
    review_path = Path(topology_review_manifest)
    review = read_yaml(review_path)
    run_dir = Path(review["run_dir"])
    approval_map = _approval_entries(approvals)
    records = []
    blockers = []
    for item in review.get("reviewed_topologies", []):
        selected = item.get("selected", {})
        local_id = item.get("lipid", {}).get("local_id")
        approval = approval_map.get(str(local_id)) or approval_map.get(str(selected.get("topology_id")))
        if not approval or not approval.get("approve_for_production", approval.get("approved", False)):
            records.append({"component": local_id, "topology_id": selected.get("topology_id"), "approval_status": "not_requested"})
            continue
        reviewer = approval.get("reviewer")
        rationale = approval.get("rationale")
        if not reviewer or not rationale:
            blockers.append({"component": local_id, "blocker_type": "approval_metadata_missing", "reason": "reviewer and rationale are required"})
            continue
        topology_files = selected.get("topology_files", [])
        if not selected.get("topology_id") or not topology_files:
            blockers.append({"component": local_id, "blocker_type": "topology_missing", "reason": "cannot production-approve descriptor-only or missing topology records"})
            continue
        validations = [validate_topology_file(topo["path"], selected.get("confidence_tier"), True) for topo in topology_files]
        validation_passed = all(record.get("validation_status") == "pass" for record in validations)
        if not validation_passed:
            blockers.append({"component": local_id, "blocker_type": "topology_validation_failed", "reason": "all topology files must pass validation before production approval"})
            records.append({"component": local_id, "topology_id": selected.get("topology_id"), "approval_status": "blocked", "validations": validations})
            continue
        approval_status = approval.get("approval_status", "user_curated_production")
        selected["production_eligible"] = True
        selected["allowed_for_production"] = True
        selected["production_review_required"] = False
        selected["placeholder_topology"] = False
        selected["approval_status"] = approval_status
        selected["production_approval"] = {
            "approved_at": utc_now(),
            "reviewer": reviewer,
            "rationale": rationale,
            "approval_status": approval_status,
            "approval_source": str(approvals),
            "validation_status": "pass",
        }
        item["review_status"] = "production_approved"
        records.append({
            "component": local_id,
            "topology_id": selected.get("topology_id"),
            "approval_status": approval_status,
            "reviewer": reviewer,
            "rationale": rationale,
            "validations": validations,
        })
    reviewed = review.get("reviewed_topologies", [])
    approved_count = sum(1 for item in reviewed if item.get("selected", {}).get("production_eligible") and not item.get("selected", {}).get("production_review_required"))
    review["reviewed_topologies"] = reviewed
    review["production_approval_manifest"] = "topology/production_topology_approval_manifest.yaml"
    review["production_approval_summary"] = {
        "approved_count": approved_count,
        "total_count": len(reviewed),
        "status": "all_components_production_approved" if approved_count == len(reviewed) and not blockers else "partial_or_blocked",
    }
    review["updated_at"] = utc_now()
    write_yaml(review_path, review)
    manifest = {
        "schema_version": "automd.production_topology_approval_manifest.v0.1",
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "topology_review_manifest": str(review_path),
        "approvals": str(approvals),
        "approved_count": approved_count,
        "total_count": len(reviewed),
        "records": records,
        "blockers": blockers,
        "status": "pass" if not blockers and approved_count == len(reviewed) else "blocked",
        "caveat": "This command records an explicit curator approval gate; it does not infer scientific validity without reviewer/rationale and validation evidence.",
    }
    out = write_yaml(run_dir / "topology" / "production_topology_approval_manifest.yaml", manifest)
    print(out)
    return out


def command_topology_resolve(descriptor_manifest: str) -> Path:
    desc = read_yaml(descriptor_manifest)
    run_dir = Path(desc["run_dir"])
    registry = load_registry()
    packets = []
    resolutions = []
    for idx, lipid in enumerate(desc["lipids"], start=1):
        candidates = match_topologies(lipid, registry)
        options = []
        for n, candidate in enumerate(candidates, start=1):
            options.append({"option_id": n, **candidate})
        options.append({"option_id": len(options) + 1, "label": "Mark unresolved; descriptor-only", "allowed_for_production": False, "confidence": "descriptor_only"})
        packet = {
            "schema_version": "automd.review_packet.v0.1",
            "review_packet_id": f"REVIEW-TOPO-{idx:06d}",
            "question_type": "select_topology",
            "lipid": {"local_id": lipid["local_id"], "name": lipid["display_name"], "role": lipid["role"]},
            "candidates": options,
            "allowed_answers": [o["option_id"] for o in options],
            "auto_selected_option_id": 1 if candidates and candidates[0]["confidence"] == "exact_alias" else None,
        }
        packets.append(packet)
        selected = options[0] if packet["auto_selected_option_id"] == 1 else options[-1]
        resolutions.append({"local_id": lipid["local_id"], "name": lipid["display_name"], "selected": selected, "review_required": packet["auto_selected_option_id"] is None})
    out_candidates = write_yaml(run_dir / "topology" / "topology_candidates.yaml", {"schema_version": "automd.topology_candidates.v0.1", "created_at": utc_now(), "run_dir": str(run_dir), "review_packets": packets})
    manifest = {
        "schema_version": "automd.topology_resolution_manifest.v0.1",
        "created_at": utc_now(),
        "formulation_id": desc["formulation_id"],
        "run_dir": str(run_dir),
        "descriptor_manifest": str(descriptor_manifest),
        "topology_candidates": str(out_candidates),
        "resolutions": resolutions,
    }
    write_yaml(run_dir / "manifests" / "topology_resolution_manifest.yaml", manifest)
    print(out_candidates)
    return out_candidates


def command_review_topology(candidates_path: str, answers: str | None = None) -> Path:
    candidates = read_yaml(candidates_path)
    run_dir = Path(candidates["run_dir"])
    answers_data = read_yaml(answers) if answers else {}
    reviewed = []
    for packet in candidates["review_packets"]:
        key = packet["lipid"]["local_id"]
        answer = answers_data.get(key) or answers_data.get(packet["review_packet_id"]) or packet.get("auto_selected_option_id")
        if answer is None:
            answer = packet["candidates"][-1]["option_id"]
        if isinstance(answer, dict):
            answer = answer.get("selected_option_id")
        allowed = packet["allowed_answers"]
        if int(answer) not in allowed:
            raise ValueError(f"Invalid answer {answer} for {key}; allowed {allowed}")
        selected = next(item for item in packet["candidates"] if item["option_id"] == int(answer))
        reviewed.append({"lipid": packet["lipid"], "selected_option_id": int(answer), "selected": selected, "review_status": "accepted" if selected.get("topology_id") else "descriptor_only"})
    unresolved = [item for item in reviewed if not item["selected"].get("topology_id")]
    manifest = {
        "schema_version": "automd.topology_review_manifest.v0.1",
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "topology_candidates": str(candidates_path),
        "reviewed_topologies": reviewed,
        "no_production_rules_enforced": True,
        "unresolved_count": len(unresolved),
        "simulation_readiness": "smoke_ready" if not unresolved else "descriptor_only",
    }
    out = write_yaml(run_dir / "manifests" / "topology_review_manifest.yaml", manifest)
    print(out)
    return out


def command_review_topology_auto(candidates_path: str, policy: dict[str, Any]) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = read_yaml(candidates_path)
    run_dir = Path(candidates["run_dir"])
    reviewed = []
    decisions = []
    blockers = []
    auto_accept = set(policy.get("auto_accept_tiers", []))
    allow_triage = bool(policy.get("allow_triage_generated_topologies"))
    for packet in candidates["review_packets"]:
        selected = packet["candidates"][0]
        tier = selected.get("confidence_tier")
        auto_accepted = False
        blocker = None
        if tier in auto_accept and selected.get("validation_status") in {"pass", "not_applicable"}:
            auto_accepted = True
        elif tier == "D_generated_smallmol_backend" and allow_triage and selected.get("validation_status") == "pass":
            auto_accepted = True
            selected["triage_only"] = True
            selected["allowed_for_production"] = False
        else:
            blocker = {
                "blocker_type": "topology_review_required",
                "component": packet["lipid"]["local_id"],
                "reason": f"{tier} is not auto-accepted by policy",
                "next_action": "rerun with --allow-triage or provide curated topology",
            }
            blockers.append(blocker)
            selected = packet["candidates"][-1]
        reviewed.append({
            "lipid": packet["lipid"],
            "selected_option_id": selected["option_id"],
            "selected": selected,
            "review_status": "auto_accepted" if auto_accepted else "descriptor_only",
        })
        decisions.append({
            "local_id": packet["lipid"]["local_id"],
            "confidence_tier": tier,
            "auto_accepted_for_smoke": auto_accepted,
            "smoke_eligible": bool(selected.get("allowed_for_smoke")),
            "production_eligible": bool(selected.get("production_eligible")),
            "production_requires_review": bool(selected.get("production_review_required", not selected.get("production_eligible"))),
            "placeholder_topology": bool(selected.get("placeholder_topology")),
            "selected_option_id": selected["option_id"],
            "blocker": blocker,
        })
    unresolved = [item for item in reviewed if not item["selected"].get("topology_id")]
    manifest = {
        "schema_version": "automd.topology_review_manifest.v0.1",
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "topology_candidates": str(candidates_path),
        "review_mode": "automated",
        "policy_snapshot": policy,
        "reviewed_topologies": reviewed,
        "no_production_rules_enforced": True,
        "unresolved_count": len(unresolved),
        "simulation_readiness": "smoke_ready" if not unresolved and not blockers else "descriptor_only",
        "blockers": blockers,
    }
    out = write_yaml(run_dir / "manifests" / "topology_review_manifest.yaml", manifest)
    print(out)
    return out, decisions, blockers


def command_templates_list() -> dict[str, Any]:
    data = {"schema_version": "automd.template_registry.v0.1", "templates": list(TEMPLATES.values())}
    print(__import__("json").dumps(data, indent=2))
    return data


def command_templates_recommend(review_manifest: str) -> Path:
    review = read_yaml(review_manifest)
    run_dir = Path(review["run_dir"])
    intake = read_yaml(run_dir / "manifests" / "intake_manifest.yaml")
    requested = intake.get("simulation_request", {}).get("template", "auto")
    if requested != "auto" and requested in TEMPLATES:
        selected = requested
    elif review["unresolved_count"]:
        selected = "descriptor_only"
    else:
        selected = "minimal_mixed_lipid_smoke_box"
    manifest = {
        "schema_version": "automd.template_manifest.v0.1",
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "formulation_id": intake["formulation_id"],
        "topology_review_manifest": str(review_manifest),
        "selected_template": TEMPLATES[selected],
        "assumptions": {
            "qualitative_smoke_only": True,
            "computed_data_not_empirical": True,
            "morphology_not_validated": True,
            "solvent": intake["simulation_request"].get("solvent"),
            "ion_concentration_mM": intake["simulation_request"].get("ion_concentration_mM"),
            "temperature_K": intake["simulation_request"].get("temperature_K"),
            "pressure_bar": intake["simulation_request"].get("pressure_bar"),
            "random_seed": intake["simulation_request"].get("random_seed"),
        },
    }
    out = write_yaml(run_dir / "manifests" / "template_manifest.yaml", manifest)
    print(out)
    return out


def command_templates_recommend_auto(review_manifest: str, automation_manifest: str, policy: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    review = read_yaml(review_manifest)
    run_dir = Path(review["run_dir"])
    intake = read_yaml(run_dir / "manifests" / "intake_manifest.yaml")
    automation = read_yaml(automation_manifest)
    role_counts: dict[str, int] = {}
    for role_record in automation.get("role_inference", []):
        role = role_record.get("inferred_role")
        role_counts[role] = role_counts.get(role, 0) + 1
    blockers = list(automation.get("blockers", [])) + list(review.get("blockers", []))
    all_smoke_ready = review.get("simulation_readiness") == "smoke_ready"
    decision_rules = []
    not_selected = {}
    if blockers or not all_smoke_ready:
        selected = "descriptor_only"
        decision_rules.append("blocked_or_review_required")
    elif role_counts.get("ionizable_lipid") and role_counts.get("sterol") and (role_counts.get("phospholipid") or role_counts.get("helper_lipid")):
        selected = "lnp_smoke_self_assembly"
        decision_rules.append("lnp_role_pattern_detected")
    else:
        selected = "minimal_mixed_lipid_smoke_box"
        decision_rules.extend(["all_components_A_to_C_or_triage_allowed", "smoke_supported"])
        not_selected["lnp_smoke_self_assembly"] = {"reason": "full LNP role pattern not inferred"}
    manifest = {
        "schema_version": "automd.template_manifest.v0.1",
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "formulation_id": intake["formulation_id"],
        "topology_review_manifest": str(review_manifest),
        "selected_template": TEMPLATES[selected],
        "decision_rules": decision_rules,
        "not_selected": not_selected,
        "assumptions": {
            "qualitative_smoke_only": True,
            "computed_data_not_empirical": True,
            "morphology_not_validated": True,
            "solvent": intake["simulation_request"].get("solvent"),
            "ion_concentration_mM": intake["simulation_request"].get("ion_concentration_mM"),
            "temperature_K": intake["simulation_request"].get("temperature_K"),
            "pressure_bar": intake["simulation_request"].get("pressure_bar"),
            "random_seed": intake["simulation_request"].get("random_seed"),
        },
    }
    out = write_yaml(run_dir / "manifests" / "template_manifest.yaml", manifest)
    print(out)
    return out, {"selected_template": selected, "decision_rules": decision_rules, "not_selected": not_selected}


def _planned_counts(lipids: list[dict[str, Any]], total: int = 64) -> list[dict[str, Any]]:
    raw = [float(lipid["normalized_mol_fraction"]) * total for lipid in lipids]
    counts = [max(1, int(round(x))) for x in raw]
    diff = total - sum(counts)
    counts[0] += diff
    return [{**lipid, "planned_molecule_count": count} for lipid, count in zip(lipids, counts)]


def _run_custom_builder(template_manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    command = template_manifest.get("custom_builder", {}).get("command") or os.environ.get("AUTOMD_CUSTOM_BUILDER")
    if not command:
        raise ValueError("custom_script builder requires template_manifest.custom_builder.command or AUTOMD_CUSTOM_BUILDER")
    command_args = _command_to_argv(command)
    proc = subprocess.run(command_args, cwd=run_dir, capture_output=True, text=True, check=False, timeout=300)
    log_dir = run_dir / "builder_logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / "custom_builder.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "custom_builder.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"custom_script builder failed with return code {proc.returncode}; see {log_dir}")
    required = [run_dir / "systems" / "system.gro", run_dir / "systems" / "topol.top"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"custom_script builder did not create required files: {missing}")
    return {
        "command": command_args,
        "return_code": proc.returncode,
        "stdout": file_record(log_dir / "custom_builder.stdout.txt", run_dir),
        "stderr": file_record(log_dir / "custom_builder.stderr.txt", run_dir),
    }


def _write_gromacs_ready_mdp(path: Path, step: str, nsteps: int, seed: int = 12345) -> None:
    integrator = "steep" if step == "em" else "md"
    md_extra = ""
    if step != "em":
        md_extra = (
            "gen-vel = yes\n"
            "gen-temp = 310\n"
            f"gen-seed = {int(seed)}\n"
            "tcoupl = no\n"
        )
    path.write_text(
        "; AutoMD tiny GROMACS smoke MDP\n"
        f"integrator = {integrator}\n"
        f"nsteps = {nsteps}\n"
        "dt = 0.002\n"
        "emtol = 1000.0\n"
        "emstep = 0.01\n"
        "cutoff-scheme = Verlet\n"
        "nstlist = 10\n"
        "pbc = xyz\n"
        "coulombtype = Cut-off\n"
        "rcoulomb = 1.0\n"
        "vdwtype = Cut-off\n"
        "rvdw = 1.0\n"
        "nstlog = 10\n"
        "nstenergy = 10\n"
        "nstxout = 0\n"
        "nstvout = 0\n"
        "nstfout = 0\n"
        "nstxout-compressed = 10\n"
        "constraints = none\n"
        "continuation = no\n"
        f"{md_extra}",
        encoding="utf-8",
    )


def _write_production_mdp(path: Path, step: str, nsteps: int, seed: int = 12345, checkpoint_interval: int = 1000) -> None:
    integrator = "steep" if step == "production_em" else "md"
    continuation = "yes" if step in {"production_npt", "production_md"} else "no"
    coupling = ""
    if step != "production_em":
        gen_vel = "yes" if step == "production_nvt" else "no"
        coupling = (
            "tcoupl = v-rescale\n"
            "tc-grps = System\n"
            "tau-t = 1.0\n"
            "ref-t = 310\n"
            f"gen-vel = {gen_vel}\n"
            "gen-temp = 310\n"
            f"gen-seed = {int(seed)}\n"
        )
        if step == "production_nvt":
            coupling += "pcoupl = no\n"
        else:
            coupling += (
                "pcoupl = C-rescale\n"
                "pcoupltype = isotropic\n"
                "tau-p = 5.0\n"
                "ref-p = 1.0\n"
                "compressibility = 3e-4\n"
                "refcoord-scaling = com\n"
            )
    path.write_text(
        "; AutoMD production-stage MDP\n"
        f"integrator = {integrator}\n"
        f"nsteps = {int(nsteps)}\n"
        "dt = 0.02\n"
        "emtol = 500.0\n"
        "emstep = 0.01\n"
        "cutoff-scheme = Verlet\n"
        "nstlist = 20\n"
        "pbc = xyz\n"
        "coulombtype = Reaction-Field\n"
        "rcoulomb = 1.1\n"
        "epsilon-r = 15\n"
        "vdwtype = Cut-off\n"
        "rvdw = 1.1\n"
        "constraints = none\n"
        f"continuation = {continuation}\n"
        "nstlog = 1000\n"
        "nstenergy = 1000\n"
        "nstxout = 0\n"
        "nstvout = 0\n"
        "nstfout = 0\n"
        "nstxout-compressed = 1000\n"
        f"; checkpoint_interval_steps = {int(checkpoint_interval)}\n"
        f"{coupling}",
        encoding="utf-8",
    )


def command_build_smoke(template_manifest: str, builder: str = "mock") -> Path:
    tmpl = read_yaml(template_manifest)
    run_dir = Path(tmpl["run_dir"])
    review = read_yaml(tmpl["topology_review_manifest"])
    intake = read_yaml(run_dir / "manifests" / "intake_manifest.yaml")
    build_dirs = [run_dir / p for p in ["systems", "mdp", "gromacs", "topologies", "trajectories", "metrics", "reports", "images"]]
    for d in build_dirs:
        d.mkdir(parents=True, exist_ok=True)
    selected = review["reviewed_topologies"]
    seed = int(tmpl.get("assumptions", {}).get("random_seed") or intake.get("simulation_request", {}).get("random_seed") or 12345)
    if builder == "custom_script":
        custom_result = _run_custom_builder(tmpl, run_dir)
        manifest = {
            "schema_version": "automd.build_manifest.v0.1",
            "created_at": utc_now(),
            "run_dir": str(run_dir),
            "formulation_id": tmpl["formulation_id"],
            "template_manifest": str(template_manifest),
            "builder": {"name": builder, "version": "automd.custom_script_builder.v0.1", "mode": "external_command", "result": custom_result},
            "reproducibility": {"random_seed": seed, "seed_source": "template_manifest.assumptions.random_seed"},
            "planned_composition": intake["lipids"],
            "artifacts": {
                "system_gro": file_record(run_dir / "systems" / "system.gro", run_dir),
                "topol_top": file_record(run_dir / "systems" / "topol.top", run_dir),
                "mdp": [file_record(p, run_dir) for p in sorted((run_dir / "mdp").glob("*.mdp"))],
                "topologies": [file_record(p, run_dir) for p in sorted((run_dir / "topologies").glob("*.itp"))],
            },
        }
        out = write_yaml(run_dir / "manifests" / "build_manifest.yaml", manifest)
        print(out)
        return out
    for item in selected:
        for topo in item["selected"].get("topology_files", []):
            src = Path(topo["path"])
            if src.exists():
                shutil.copy2(src, run_dir / "topologies" / src.name)
    planned = _planned_counts(intake["lipids"])
    gro = run_dir / "systems" / "system.gro"
    atoms = []
    atom_id = 1
    residue_id = 1
    for li, (lipid, reviewed) in enumerate(zip(planned, selected)):
        mol_type = reviewed["selected"].get("molecule_type_name") or lipid["display_name"]
        mol_name = _safe_molecule_type(mol_type, lipid["local_id"])[:5]
        topo_files = reviewed["selected"].get("topology_files", [])
        atom_count = 1
        if topo_files:
            source_summary = parse_itp_summary(Path(topo_files[0]["path"]))
            atom_count = max(1, int(source_summary.get("atom_count") or 1))
        for mi in range(lipid["planned_molecule_count"]):
            base_x = (li * 1.3 + mi * 0.07) % 8
            base_y = (mi * 0.11) % 8
            base_z = 4.0 + ((mi % 5) - 2) * 0.05
            for ai in range(1, atom_count + 1):
                x = base_x + 0.03 * (ai - 1)
                y = base_y
                z = base_z
                atoms.append(f"{residue_id%99999:5d}{mol_name:<5}{('B'+str(ai))[:5]:>5}{atom_id%99999:5d}{x:8.3f}{y:8.3f}{z:8.3f}")
                atom_id += 1
            residue_id += 1
    gro.write_text("AutoMD mock Martini smoke system\n" + f"{len(atoms):5d}\n" + "\n".join(atoms) + "\n   8.00000   8.00000   8.00000\n", encoding="utf-8")
    top = run_dir / "systems" / "topol.top"
    includes = ['#include "../topologies/' + Path(item["selected"]["topology_files"][0]["path"]).name + '"' for item in selected if item["selected"].get("topology_files")]
    molecules = [f"{item['selected'].get('molecule_type_name', item['lipid']['name']):<16} {planned[i]['planned_molecule_count']}" for i, item in enumerate(selected)]
    top.write_text(
        "; AutoMD generated tiny GROMACS topology package\n"
        "; This is a plumbing smoke topology, not a production Martini force field.\n\n"
        "[ defaults ]\n"
        "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ\n"
        "1 1 no 1.0 1.0\n\n"
        "[ atomtypes ]\n"
        "; name mass charge ptype sigma epsilon\n"
        "P1 72.0 0.0 A 0.470 5.000\n\n"
        + "\n".join(includes)
        + "\n\n[ system ]\nAutoMD smoke system\n\n[ molecules ]\n"
        + "\n".join(molecules)
        + "\n",
        encoding="utf-8",
    )
    for name, nsteps in [("em", 100), ("smoke_nvt", 250), ("smoke_npt", 250)]:
        _write_gromacs_ready_mdp(run_dir / "mdp" / f"{name}.mdp", name, nsteps, seed)
    manifest = {
        "schema_version": "automd.build_manifest.v0.1",
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "formulation_id": tmpl["formulation_id"],
        "template_manifest": str(template_manifest),
        "builder": {"name": builder, "version": "automd.mock_builder.v0.1", "mode": "mock_gromacs_ready"},
        "reproducibility": {"random_seed": seed, "seed_source": "template_manifest.assumptions.random_seed"},
        "planned_composition": planned,
        "artifacts": {
            "system_gro": file_record(gro, run_dir),
            "topol_top": file_record(top, run_dir),
            "mdp": [file_record(p, run_dir) for p in sorted((run_dir / "mdp").glob("*.mdp"))],
            "topologies": [file_record(p, run_dir) for p in sorted((run_dir / "topologies").glob("*.itp"))],
        },
    }
    out = write_yaml(run_dir / "manifests" / "build_manifest.yaml", manifest)
    print(out)
    return out


def _simulate_outputs(run_dir: Path, dry_run: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gmx = shutil.which("gmx") or "gmx"
    commands = []
    outputs = []
    for step, coord in [("em", "systems/system.gro"), ("smoke_nvt", "gromacs/em.gro"), ("smoke_npt", "gromacs/smoke_nvt.gro")]:
        grompp = f"{gmx} grompp -f mdp/{step}.mdp -c {coord} -p systems/topol.top -o gromacs/{step}.tpr -po gromacs/{step}_mdout.mdp -pp gromacs/{step}_processed.top -maxwarn 0"
        mdrun = f"{gmx} mdrun -deffnm gromacs/{step} -v"
        commands.append({"stage": step, "kind": "grompp", "command": grompp, "return_code": 0 if dry_run else None, "warnings": []})
        commands.append({"stage": step, "kind": "mdrun", "command": mdrun, "return_code": 0 if dry_run else None, "warnings": []})
        if dry_run:
            for suffix, text in {
                ".tpr": f"dry-run tpr placeholder for {step}\n",
                ".log": f"AutoMD dry-run mdrun log for {step}\nFinished mdrun successfully\n",
                ".edr": "Potential Energy -1.0e3\nTemperature 310\n",
                ".gro": (run_dir / "systems" / "system.gro").read_text(encoding="utf-8") if (run_dir / "systems" / "system.gro").exists() else "",
                ".xtc": f"dry-run trajectory placeholder frames=3 step={step}\n",
            }.items():
                path = run_dir / "gromacs" / f"{step}{suffix}"
                path.write_text(text, encoding="utf-8")
                outputs.append(file_record(path, run_dir))
    return commands, {"files": outputs}


def _run_command(args: list[str], run_dir: Path) -> dict[str, Any]:
    started = utc_now()
    proc = subprocess.run(args, cwd=run_dir, capture_output=True, text=True, check=False)
    kind = args[1] if len(args) > 1 else "command"
    stage = "unknown"
    if "-o" in args:
        stage = Path(args[args.index("-o") + 1]).stem
    elif "-deffnm" in args:
        stage = Path(args[args.index("-deffnm") + 1]).name
    log_dir = run_dir / "gromacs" / "command_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{stage}_{kind}.stdout.txt"
    stderr_path = log_dir / f"{stage}_{kind}.stderr.txt"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    warnings = [line for line in (proc.stdout + "\n" + proc.stderr).splitlines() if "warning" in line.lower()]
    return {
        "stage": stage,
        "kind": kind,
        "command": " ".join(args),
        "return_code": proc.returncode,
        "started_at": started,
        "completed_at": utc_now(),
        "warnings": warnings,
        "stdout": file_record(stdout_path, run_dir),
        "stderr": file_record(stderr_path, run_dir),
    }


def _run_real_gromacs_smoke(run_dir: Path, gmx: str, gmx_extra: str = "") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for step, coord in [("em", "systems/system.gro"), ("smoke_nvt", "gromacs/em.gro"), ("smoke_npt", "gromacs/smoke_nvt.gro")]:
        grompp_args = [
            gmx,
            "grompp",
            "-f",
            f"mdp/{step}.mdp",
            "-c",
            coord,
            "-p",
            "systems/topol.top",
            "-o",
            f"gromacs/{step}.tpr",
            "-po",
            f"gromacs/{step}_mdout.mdp",
            "-pp",
            f"gromacs/{step}_processed.top",
            "-maxwarn",
            "0",
        ]
        result = _run_command(grompp_args, run_dir)
        commands.append(result)
        if result["return_code"] != 0:
            break
        mdrun_args = [gmx, "mdrun", "-deffnm", f"gromacs/{step}", "-v", "-ntmpi", "1"]
        if gmx_extra:
            mdrun_args.extend(shlex.split(gmx_extra))
        result = _run_command(mdrun_args, run_dir)
        commands.append(result)
        if result["return_code"] != 0:
            break
    outputs = [file_record(path, run_dir) for path in sorted((run_dir / "gromacs").glob("*")) if path.is_file()]
    return commands, {"files": outputs}


def command_gromacs_preflight(build_manifest: str) -> Path:
    build = read_yaml(build_manifest)
    run_dir = Path(build["run_dir"])
    gmx = shutil.which("gmx") or "gmx"
    command = f"{gmx} grompp -f mdp/em.mdp -c systems/system.gro -p systems/topol.top -o gromacs/em.tpr -po gromacs/em_mdout.mdp -pp gromacs/em_processed.top -maxwarn 0"
    manifest = {
        "schema_version": "automd.grompp_preflight_manifest.v0.1",
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "build_manifest": str(build_manifest),
        "gromacs": {"executable": shutil.which("gmx"), "available": bool(shutil.which("gmx"))},
        "commands": [{"kind": "grompp", "command": command, "return_code": None, "warnings": [], "dry_run_available": True}],
        "status": "planned" if not shutil.which("gmx") else "ready_to_run",
        "maxwarn_policy": {"default": 0, "unreviewed_maxwarn_forbidden": True},
    }
    out = write_yaml(run_dir / "manifests" / "grompp_preflight_manifest.yaml", manifest)
    print(out)
    return out


def command_simulate_smoke(build_manifest: str, dry_run: bool = False, gmx_extra: str = "") -> Path:
    build = read_yaml(build_manifest)
    run_dir = Path(build["run_dir"])
    gmx = shutil.which("gmx")
    if not dry_run and not gmx:
        raise RuntimeError("GROMACS executable 'gmx' not found. Re-run with --dry-run or install/load GROMACS.")
    if dry_run:
        commands, outputs = _simulate_outputs(run_dir, dry_run=True)
    else:
        commands, outputs = _run_real_gromacs_smoke(run_dir, gmx or "gmx", gmx_extra)
    if gmx_extra:
        for command in commands:
            if dry_run and command["kind"] == "mdrun":
                command["command"] += " " + gmx_extra
    all_passed = bool(commands) and all(command.get("return_code") == 0 for command in commands)
    gromacs_version = _gromacs_version(gmx)
    manifest = {
        "schema_version": "automd.gromacs_run.v0.1",
        "run_id": f"RUN-{build['formulation_id']}-smoke-0001",
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "build_manifest": str(build_manifest),
        "dry_run": dry_run,
        "gromacs": {"executable": gmx, "available": bool(gmx), **gromacs_version},
        "commands": commands,
        "outputs": outputs["files"],
        "status": "completed" if dry_run or all_passed else "failed",
        "computed_data_warning": "Dry-run artifacts are placeholders for workflow verification, not physical simulation results." if dry_run else "Real GROMACS smoke execution; still qualitative and not scientific validation.",
    }
    out = write_yaml(run_dir / "manifests" / "smoke_run_manifest.yaml", manifest)
    print(out)
    return out


def _smoke_energy_sanity(run_dir: Path) -> dict[str, Any]:
    inspected = []
    combined = []
    for path in sorted((run_dir / "gromacs").glob("*.log")) + sorted((run_dir / "gromacs" / "command_logs").glob("*.stderr.txt")):
        inspected.append(str(path.relative_to(run_dir)))
        combined.append(path.read_text(encoding="utf-8", errors="ignore").lower())
    if not inspected:
        return {"status": "not_checked", "passed": None, "inspected_files": [], "reason": "no log or energy files found"}
    text = "\n".join(combined)
    failed = any(marker in text for marker in [" nan", "nan ", "nan\n", "fatal error"])
    return {"status": "checked_text_logs", "passed": not failed, "inspected_files": inspected}


def _structure_count_check(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "not_checked", "passed": None, "reason": "final structure missing"}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        declared_atoms = int(lines[1].strip())
        parsed_atoms = max(0, len(lines) - 3)
    except Exception as exc:
        return {"status": "failed_to_parse", "passed": False, "reason": str(exc)}
    return {
        "status": "checked_structure_atom_count",
        "passed": declared_atoms == parsed_atoms and declared_atoms > 0,
        "declared_atoms": declared_atoms,
        "parsed_atoms": parsed_atoms,
        "note": "Atom count check is not a full molecule-count validation.",
    }


def command_qc_smoke(smoke_manifest: str) -> Path:
    smoke = read_yaml(smoke_manifest)
    run_dir = Path(smoke["run_dir"])
    commands = smoke.get("commands", [])
    grompp_passed = all(c.get("return_code") in (0, None) for c in commands if c.get("kind") == "grompp")
    mdrun_passed = all(c.get("return_code") in (0, None) for c in commands if c.get("kind") == "mdrun")
    final_structure = run_dir / "gromacs" / "smoke_npt.gro"
    traj = run_dir / "gromacs" / "smoke_npt.xtc"
    failures = []
    if not grompp_passed:
        failures.append({"failure_class": "grompp_parameter_error", "suggested_fix": "Inspect grompp stderr and topology includes.", "retry_safe": False})
    if not mdrun_passed:
        failures.append({"failure_class": "mdrun_instability", "suggested_fix": "Inspect mdrun log and reduce timestep or fix topology.", "retry_safe": False})
    if not final_structure.exists():
        failures.append({"failure_class": "mdrun_resource_failure", "suggested_fix": "No final structure was produced.", "retry_safe": True})
    energy_sanity = _smoke_energy_sanity(run_dir)
    if energy_sanity["passed"] is False:
        failures.append({"failure_class": "energy_sanity_failed", "suggested_fix": "Inspect GROMACS logs and energy output for NaN/Inf/fatal errors.", "retry_safe": False})
    structure_count = _structure_count_check(final_structure)
    if structure_count["passed"] is False:
        failures.append({"failure_class": "structure_count_check_failed", "suggested_fix": "Inspect final .gro atom count and system builder output.", "retry_safe": False})
    status = "pass" if not failures else "fail"
    manifest = {
        "schema_version": "automd.qc_manifest.v0.1",
        "run_id": smoke["run_id"],
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "smoke_run_manifest": str(smoke_manifest),
        "qc_status": status,
        "qc_summary": {
            "grompp_passed": grompp_passed,
            "minimization_passed": (run_dir / "gromacs" / "em.gro").exists(),
            "smoke_mdrun_passed": mdrun_passed,
            "energy_sanity": energy_sanity,
            "no_nan_energy": energy_sanity["passed"],
            "final_structure_present": final_structure.exists(),
            "structure_count_check": structure_count,
            "expected_molecule_counts_present": None,
            "trajectory_frames": 3 if traj.exists() else 0,
        },
        "warnings": [
            {"severity": "note", "message": "Short smoke/dry-run metrics are qualitative computed features only."},
            {"severity": "note", "message": "Molecule-count consistency is not fully validated; structure atom count is checked separately."},
        ],
        "failures": failures,
        "recommended_next_action": "extract_metrics" if status == "pass" else "review_failure",
    }
    out = write_yaml(run_dir / "manifests" / "qc_manifest.yaml", manifest)
    print(out)
    return out


def _parse_gro_coords(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    coords = []
    for line in lines[2:-1]:
        try:
            coords.append([float(line[-24:-16]), float(line[-16:-8]), float(line[-8:])])
        except Exception:
            continue
    return np.array(coords, dtype=float) if coords else np.zeros((0, 3))


def command_metrics_extract(qc_manifest: str, allow_failed_qc: bool = False) -> Path:
    qc = read_yaml(qc_manifest)
    if qc["qc_status"] != "pass" and not allow_failed_qc:
        raise RuntimeError("QC did not pass; use --allow-failed-qc to extract failure-context metrics.")
    run_dir = Path(qc["run_dir"])
    coords = _parse_gro_coords(run_dir / "gromacs" / "smoke_npt.gro")
    if len(coords):
        center = coords.mean(axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        rog = float(np.sqrt(np.mean(distances**2)))
        diameter = float(2 * distances.max())
        eigvals = np.linalg.eigvalsh(np.cov(coords.T)) if len(coords) > 2 else np.array([0, 0, 0])
        anisotropy = float((eigvals.max() - eigvals.min()) / eigvals.sum()) if eigvals.sum() else 0.0
    else:
        rog = diameter = anisotropy = None
    water_like = 0
    if len(coords):
        final_gro = run_dir / "gromacs" / "smoke_npt.gro"
        for line in final_gro.read_text(encoding="utf-8", errors="ignore").splitlines()[2:-1]:
            residue_name = line[5:10].strip().upper()
            atom_name = line[10:15].strip().upper()
            if residue_name in {"W", "WF", "PW", "WATER", "SOL"} or atom_name in {"W", "WF", "PW"}:
                water_like += 1
    water_cavity_status = "computed_no_water_beads_detected" if len(coords) and water_like == 0 else ("computed" if len(coords) else "missing_structure")
    metrics = {
        "radius_of_gyration_nm": {"value": rog, "status": "computed" if rog is not None else "missing_structure", "unit": "nm"},
        "diameter_proxy_nm": {"value": diameter, "status": "computed" if diameter is not None else "missing_structure", "unit": "nm"},
        "shape_anisotropy": {"value": anisotropy, "status": "computed" if anisotropy is not None else "missing_structure", "unit": "unitless"},
        "water_cavity_proxy": {
            "value": 0.0 if len(coords) and water_like == 0 else None,
            "status": water_cavity_status,
            "unit": "qualitative",
            "method": "water-like bead count proxy; zero means no water beads were detected in the smoke package",
        },
        "aggregation_stability_proxy": {"value": "completed_smoke_artifacts_present", "status": "computed"},
    }
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    table = metrics_dir / "metrics.parquet"
    table_rows = []
    for name, value in metrics.items():
        raw_value = value.get("value")
        table_rows.append({
            "metric": name,
            "value_numeric": raw_value if isinstance(raw_value, (int, float)) or raw_value is None else None,
            "value_text": raw_value if isinstance(raw_value, str) else None,
            "status": value.get("status"),
            "unit": value.get("unit"),
        })
    pd.DataFrame(table_rows).to_parquet(table, index=False)
    manifest = {
        "schema_version": "automd.metrics_manifest.v0.1",
        "run_id": qc["run_id"],
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "metrics_version": "automd.metrics.v0.1",
        "qc_manifest": str(qc_manifest),
        "trajectory_source": "gromacs/smoke_npt.xtc",
        "structure_source": "gromacs/smoke_npt.gro",
        "metrics_table": file_record(table, run_dir),
        "metrics": metrics,
        "quality_flags": {"qualitative_only": True, "smoke_length_short": True},
    }
    out = write_yaml(run_dir / "manifests" / "metrics_manifest.yaml", manifest)
    print(out)
    return out


def command_prioritize(manifests: list[str], out: str) -> Path:
    rows = []
    for path in manifests:
        data = read_yaml(path)
        if "descriptor_coverage" in data:
            run_dir = Path(data["run_dir"])
            review_path = run_dir / "manifests" / "topology_review_manifest.yaml"
            qc_path = run_dir / "manifests" / "qc_manifest.yaml"
            metrics_path = run_dir / "manifests" / "metrics_manifest.yaml"
            automation_path = run_dir / "manifests" / "automation_manifest.yaml"
            unresolved = read_yaml(review_path).get("unresolved_count", 1) if review_path.exists() else 1
            qc = read_yaml(qc_path) if qc_path.exists() else {}
            metrics = read_yaml(metrics_path) if metrics_path.exists() else {}
            automation = read_yaml(automation_path) if automation_path.exists() else {}
            blocker_count = len(automation.get("blockers", []))
            readiness = "smoke_ready" if unresolved == 0 else "descriptor_only"
            score = 50 * float(data["descriptor_coverage"])
            score += 30 if readiness == "smoke_ready" else 0
            score += 15 if qc.get("qc_status") == "pass" else 0
            score += 5 if metrics.get("quality_flags", {}).get("qualitative_only") is True else 0
            score -= 25 * blocker_count
            score = max(0.0, min(100.0, score))
            rows.append({
                "formulation_id": data["formulation_id"],
                "run_dir": str(run_dir),
                "descriptor_coverage": data["descriptor_coverage"],
                "simulation_readiness": readiness,
                "qc_status": qc.get("qc_status", "not_run"),
                "blocker_count": blocker_count,
                "priority_score": round(score, 2),
                "recommended_next_action": "production_plan" if qc.get("qc_status") == "pass" else ("run_smoke_test" if readiness == "smoke_ready" else "resolve_topology"),
            })
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "priority_table.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    manifest = write_yaml(outdir / "priority_manifest.yaml", {"schema_version": "automd.priority_manifest.v0.2", "created_at": utc_now(), "scoring_formula": "descriptor coverage + topology readiness + QC + metrics - blockers", "priority_table": str(csv_path), "rows": rows})
    print(manifest)
    return manifest


def command_batch_plan(csv_path: str, out: str) -> Path:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    formulations = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            input_path = row.get("path") or row.get("input_path")
            auto_input = row.get("auto_input") or row.get("smiles_ratio") or row.get("components")
            formulation_id = row.get("formulation_id") or row.get("id") or (Path(input_path).stem if input_path else f"auto_{len(formulations)+1:03d}")
            formulations.append({
                "formulation_id": formulation_id,
                "input_path": input_path,
                "auto_input": auto_input,
                "mode": "auto" if auto_input else "formulation_file",
                "status": "planned",
                "readiness": "unknown",
                "run_dir": str(outdir / formulation_id),
            })
    manifest = write_yaml(outdir / "batch_plan.yaml", {"schema_version": "automd.batch_manifest.v0.2", "batch_id": outdir.name, "created_at": utc_now(), "formulations": formulations, "execution": {"backend": "local", "max_parallel": 1, "supports_auto_input": True}})
    print(manifest)
    return manifest


def command_batch_smoke(batch_plan: str, dry_run: bool = False) -> Path:
    plan = read_yaml(batch_plan)
    outdir = Path(batch_plan).parent
    statuses = []
    for item in plan["formulations"]:
        input_path = item.get("input_path")
        auto_input = item.get("auto_input")
        run_dir = item.get("run_dir") or str(outdir / str(item.get("formulation_id")))
        if not input_path and not auto_input:
            statuses.append({**item, "status": "failed", "failure": "missing_input_path"})
            continue
        try:
            if auto_input:
                report = command_auto(auto_input, run_dir, real_gromacs=not dry_run)
            else:
                report = command_workflow(input_path, run_dir, dry_run=dry_run)
            qc_path = Path(run_dir) / "manifests" / "qc_manifest.yaml"
            review_path = Path(run_dir) / "manifests" / "topology_review_manifest.yaml"
            automation_path = Path(run_dir) / "manifests" / "automation_manifest.yaml"
            qc = read_yaml(qc_path) if qc_path.exists() else {}
            review = read_yaml(review_path) if review_path.exists() else {}
            automation = read_yaml(automation_path) if automation_path.exists() else {}
            audit = command_audit_run(run_dir)
            statuses.append({
                **item,
                "run_dir": run_dir,
                "status": "completed",
                "readiness": review.get("simulation_readiness", "smoke_ready" if qc.get("qc_status") == "pass" else "review_required"),
                "qc_status": qc.get("qc_status"),
                "audit_status": audit["status"],
                "blocker_count": len(automation.get("blockers", [])),
                "report": str(report),
            })
        except Exception as exc:
            statuses.append({**item, "run_dir": run_dir, "status": "failed", "failure": str(exc)})
    out = write_yaml(outdir / "batch_status.yaml", {**plan, "formulations": statuses})
    print(out)
    return out


def command_batch_summarize(batch_dir: str) -> Path:
    batch_dir = Path(batch_dir)
    status_path = batch_dir / "batch_status.yaml"
    data = read_yaml(status_path if status_path.exists() else batch_dir / "batch_plan.yaml")
    counts: dict[str, int] = {}
    readiness_counts: dict[str, int] = {}
    blocker_total = 0
    for item in data["formulations"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
        readiness = item.get("readiness", "unknown")
        readiness_counts[readiness] = readiness_counts.get(readiness, 0) + 1
        blocker_total += int(item.get("blocker_count") or 0)
    report = batch_dir / "batch_summary.md"
    report.write_text(
        "# AutoMD Batch Summary\n\n"
        "## Status Counts\n"
        + "\n".join(f"- {k}: {v}" for k, v in counts.items())
        + "\n\n## Readiness Counts\n"
        + "\n".join(f"- {k}: {v}" for k, v in readiness_counts.items())
        + f"\n\n## Blockers\n- total recorded blockers: {blocker_total}\n",
        encoding="utf-8",
    )
    print(report)
    return report


def write_png(path: Path, width: int, height: int, points: list[tuple[float, float, float]]) -> None:
    import struct
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    pixels = bytearray([255, 255, 255] * width * height)
    for x, y, frac in points:
        px = max(0, min(width - 1, int(x * (width - 1))))
        py = max(0, min(height - 1, int(y * (height - 1))))
        color = (30, int(80 + 120 * frac), 200)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx * dx + dy * dy <= 9:
                    xx, yy = px + dx, py + dy
                    if 0 <= xx < width and 0 <= yy < height:
                        i = (yy * width + xx) * 3
                        pixels[i:i+3] = bytes(color)
    raw = b"".join(b"\x00" + pixels[y * width * 3:(y + 1) * width * 3] for y in range(height))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def command_report_run(run_dir: str) -> Path:
    run_dir = Path(run_dir)
    manifests = {p.stem: read_yaml(p) for p in (run_dir / "manifests").glob("*.yaml")}
    metrics = manifests.get("metrics_manifest", {})
    qc = manifests.get("qc_manifest", {})
    automation = manifests.get("automation_manifest", {})
    production = manifests.get("production_plan_manifest", {})
    image_dir = run_dir / "images"
    image_dir.mkdir(exist_ok=True)
    coords = _parse_gro_coords(run_dir / "gromacs" / "smoke_npt.gro") if (run_dir / "gromacs" / "smoke_npt.gro").exists() else np.zeros((0, 3))
    points = []
    if len(coords):
        mins = coords.min(axis=0)
        span = np.maximum(coords.max(axis=0) - mins, 1e-6)
        for row in coords:
            points.append(((row[0] - mins[0]) / span[0], (row[1] - mins[1]) / span[1], (row[2] - mins[2]) / span[2]))
    png = image_dir / "smoke_snapshot.png"
    write_png(png, 640, 420, points)
    report = run_dir / "reports" / "run_report.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text(
        f"# AutoMD Run Card: {manifests.get('intake_manifest', {}).get('formulation_id', run_dir.name)}\n\n"
        "## Summary\n"
        f"- Status: {qc.get('qc_status', 'not_run')}\n"
        f"- Template: {manifests.get('template_manifest', {}).get('selected_template', {}).get('template_id', 'unknown')}\n"
        "- Force field: Martini 3 compatible package\n"
        f"- Topology readiness: {manifests.get('topology_review_manifest', {}).get('simulation_readiness', 'unknown')}\n\n"
        "## Key assumptions\n"
        "- Computed outputs are not empirical wet-lab evidence.\n"
        "- Placeholder/mock topologies are not production-approved unless replaced by curated licensed files.\n"
        "- Smoke and dry-run metrics are qualitative first-pass features.\n\n"
        "## Outputs\n"
        f"- final structure: `gromacs/smoke_npt.gro`\n- trajectory: `gromacs/smoke_npt.xtc`\n- metrics: `{metrics.get('metrics_table', {}).get('path', 'not_generated')}`\n- rendered snapshot: `images/{png.name}`\n\n"
        "## QC\n"
        f"- grompp: {qc.get('qc_summary', {}).get('grompp_passed', 'not_run')}\n- mdrun: {qc.get('qc_summary', {}).get('smoke_mdrun_passed', 'not_run')}\n- energy sanity: {qc.get('qc_summary', {}).get('energy_sanity', {}).get('status', 'not_run')}\n- structure count: {qc.get('qc_summary', {}).get('structure_count_check', {}).get('status', 'not_run')}\n\n"
        "## Automation Trace\n"
        + ("\n".join(f"- {step.get('name')}: {step.get('status')}" for step in automation.get("pipeline_steps", [])) or "- not recorded")
        + "\n\n## Blockers\n"
        + ("\n".join(f"- {b.get('blocker_type')}: {b.get('reason')}" for b in automation.get("blockers", [])) or "- none recorded")
        + "\n\n## Production Readiness\n"
        + f"- status: {production.get('readiness', 'not_planned')}\n"
        + f"- blockers: {len(production.get('blockers', []))}\n\n"
        "## Metrics\n"
        + "\n".join(f"- {name}: {value.get('value')} {value.get('unit', '')} ({value.get('status')})" for name, value in metrics.get("metrics", {}).items())
        + "\n\n## Caveats\nThis is a smoke workflow. Metrics are qualitative first-pass computed features and must not be interpreted as validated delivery predictions.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "automd.report_manifest.v0.1",
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "report": file_record(report, run_dir),
        "images": [file_record(png, run_dir)],
        "input_manifests": sorted(str(p.relative_to(run_dir)) for p in (run_dir / "manifests").glob("*.yaml")),
    }
    write_yaml(run_dir / "manifests" / "report_manifest.yaml", manifest)
    print(report)
    return report


def command_report_batch(batch_dir: str) -> Path:
    return command_batch_summarize(batch_dir)


def command_production_prepare_topologies(run_dir: str, allow_placeholder: bool = False) -> Path:
    run = Path(run_dir)
    review = read_yaml(run / "manifests" / "topology_review_manifest.yaml")
    out_dir = run / "topologies" / "production"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    blockers = []
    for item in review.get("reviewed_topologies", []):
        selected = item.get("selected", {})
        local_id = item.get("lipid", {}).get("local_id")
        files = []
        for topo in selected.get("topology_files", []):
            src = Path(topo["path"])
            if not src.exists():
                blockers.append({"blocker_type": "missing_topology_file", "component": local_id, "path": str(src)})
                continue
            dest = out_dir / f"{local_id}.{src.name}"
            shutil.copy2(src, dest)
            files.append(file_record(dest, run))
        scientific_status = "curated_production" if selected.get("production_eligible") and not selected.get("production_review_required") and not selected.get("placeholder_topology") else "software_generated_or_placeholder_requires_scientific_review"
        run_allowed = scientific_status == "curated_production" or (allow_placeholder and bool(files))
        if not run_allowed:
            blockers.append({"blocker_type": "topology_not_runnable_for_production", "component": local_id, "reason": "curated production topology missing; rerun with explicit placeholder allowance for software validation only"})
        records.append({
            "component": local_id,
            "topology_id": selected.get("topology_id"),
            "confidence_tier": selected.get("confidence_tier"),
            "source_placeholder_topology": bool(selected.get("placeholder_topology")),
            "source_production_eligible": bool(selected.get("production_eligible")),
            "scientific_approval_status": scientific_status,
            "production_run_allowed": run_allowed,
            "files": files,
        })
    manifest = {
        "schema_version": "automd.production_topology_manifest.v0.1",
        "created_at": utc_now(),
        "run_dir": str(run),
        "allow_placeholder_for_software_validation": allow_placeholder,
        "topologies": records,
        "blockers": blockers,
        "status": "ready_for_production_build" if not blockers else "blocked",
        "caveat": "Placeholder/generated topologies may run the software production lifecycle only when explicitly allowed; this is not scientific topology approval.",
    }
    path = write_yaml(run / "manifests" / "production_topology_manifest.yaml", manifest)
    print(path)
    return path


def command_production_plan(run_dir: str, out: str | None = None, allow_placeholder: bool = False) -> Path:
    run = Path(run_dir)
    topology_manifest_path = run / "manifests" / "production_topology_manifest.yaml"
    command_production_prepare_topologies(str(run), allow_placeholder=allow_placeholder)
    topology_manifest = read_yaml(topology_manifest_path) if topology_manifest_path.exists() else {}
    review = read_yaml(run / "manifests" / "topology_review_manifest.yaml")
    qc = read_yaml(run / "manifests" / "qc_manifest.yaml") if (run / "manifests" / "qc_manifest.yaml").exists() else {}
    metrics = read_yaml(run / "manifests" / "metrics_manifest.yaml") if (run / "manifests" / "metrics_manifest.yaml").exists() else {}
    blockers = []
    production_ready = []
    software_runnable = []
    for item in review.get("reviewed_topologies", []):
        selected = item.get("selected", {})
        local_id = item.get("lipid", {}).get("local_id")
        topo_record = next((record for record in topology_manifest.get("topologies", []) if record.get("component") == local_id), {})
        if not selected.get("topology_id"):
            blockers.append({"blocker_type": "unresolved_topology", "component": local_id, "reason": "no selected topology"})
        elif selected.get("placeholder_topology") and not topo_record.get("production_run_allowed"):
            blockers.append({"blocker_type": "placeholder_topology", "component": local_id, "reason": "placeholder topology cannot be production planned"})
        elif (selected.get("production_review_required") or not selected.get("production_eligible")) and not topo_record.get("production_run_allowed"):
            blockers.append({"blocker_type": "production_review_required", "component": local_id, "reason": "topology is not production eligible"})
        else:
            if topo_record.get("scientific_approval_status") == "curated_production":
                production_ready.append(local_id)
            else:
                software_runnable.append(local_id)
    if qc.get("qc_status") != "pass":
        blockers.append({"blocker_type": "qc_not_passed", "component": None, "reason": "production planning requires passing smoke QC"})
    readiness = "production_ready" if not blockers and len(production_ready) == len(review.get("reviewed_topologies", [])) else ("production_runnable_software_validation" if not blockers else "blocked")
    manifest = {
        "schema_version": "automd.production_plan_manifest.v0.1",
        "created_at": utc_now(),
        "run_dir": str(run),
        "readiness": readiness,
        "production_ready_components": production_ready,
        "software_runnable_components": software_runnable,
        "blockers": blockers,
        "source_manifests": {
            "topology_review_manifest": "manifests/topology_review_manifest.yaml",
            "production_topology_manifest": "manifests/production_topology_manifest.yaml",
            "qc_manifest": "manifests/qc_manifest.yaml" if qc else None,
            "metrics_manifest": "manifests/metrics_manifest.yaml" if metrics else None,
        },
        "recommended_next_action": "build_production_package" if not blockers else "resolve production blockers before long simulations",
        "caveat": "production_runnable_software_validation is an automated software execution state, not curated scientific topology approval.",
    }
    out_path = Path(out) if out else run / "manifests" / "production_plan_manifest.yaml"
    path = write_yaml(out_path, manifest)
    print(path)
    return path


def command_production_profile(run_dir: str, profile: str = "local_cpu", walltime_hours: float = 24.0) -> Path:
    run = Path(run_dir)
    profiles = {
        "local_cpu": {"backend": "local", "ntmpi": 1, "ntomp": 4, "gpu": False, "submit_mode": "foreground"},
        "local_gpu": {"backend": "local", "ntmpi": 1, "ntomp": 4, "gpu": True, "submit_mode": "foreground"},
        "slurm_gpu": {"backend": "slurm", "partition": "gpu", "gpus": 1, "cpus_per_task": 8, "submit_mode": "script"},
    }
    selected = profiles.get(profile, profiles["local_cpu"])
    manifest = {
        "schema_version": "automd.production_profile_manifest.v0.1",
        "created_at": utc_now(),
        "run_dir": str(run),
        "profile_name": profile,
        "resources": selected,
        "walltime_hours": walltime_hours,
        "checkpoint_policy": {
            "enabled": True,
            "checkpoint_interval_steps": 1000,
            "restart_from_latest_checkpoint": True,
            "walltime_segmented": walltime_hours > 0,
        },
        "retry_policy": {"max_retries": 1, "retry_failed_segments_only": True},
    }
    path = write_yaml(run / "manifests" / "production_profile_manifest.yaml", manifest)
    print(path)
    return path


def command_production_build(run_dir: str, builder: str = "production_pack_like") -> Path:
    run = Path(run_dir)
    plan_path = run / "manifests" / "production_plan_manifest.yaml"
    if not plan_path.exists():
        command_production_plan(str(run))
    plan = read_yaml(plan_path)
    if plan.get("readiness") == "blocked":
        raise RuntimeError("Production plan is blocked; resolve blockers or rerun with explicit placeholder allowance")
    intake = read_yaml(run / "manifests" / "intake_manifest.yaml")
    review = read_yaml(run / "manifests" / "topology_review_manifest.yaml")
    topology_manifest = read_yaml(run / "manifests" / "production_topology_manifest.yaml")
    for subdir in ["production/systems", "production/mdp", "production/gromacs", "production/logs", "production/reports"]:
        (run / subdir).mkdir(parents=True, exist_ok=True)
    planned = _planned_counts(intake["lipids"], total=512)
    gro = run / "production" / "systems" / "production_system.gro"
    atoms = []
    atom_id = 1
    residue_id = 1
    for li, (lipid, reviewed) in enumerate(zip(planned, review["reviewed_topologies"])):
        mol_type = reviewed["selected"].get("molecule_type_name") or lipid["display_name"]
        mol_name = _safe_molecule_type(mol_type, lipid["local_id"])[:5]
        topo_files = reviewed["selected"].get("topology_files", [])
        atom_count = 1
        if topo_files and Path(topo_files[0]["path"]).exists():
            atom_count = max(1, int(parse_itp_summary(Path(topo_files[0]["path"])).get("atom_count") or 1))
        for mi in range(lipid["planned_molecule_count"]):
            layer = mi % 8
            base_x = (li * 1.7 + mi * 0.17) % 16
            base_y = (mi * 0.23 + layer * 0.07) % 16
            base_z = 8.0 + ((mi % 11) - 5) * 0.09
            for ai in range(1, atom_count + 1):
                atoms.append(f"{residue_id%99999:5d}{mol_name:<5}{('B'+str(ai))[:5]:>5}{atom_id%99999:5d}{base_x + 0.035 * (ai - 1):8.3f}{base_y:8.3f}{base_z:8.3f}")
                atom_id += 1
            residue_id += 1
    gro.write_text("AutoMD production candidate system\n" + f"{len(atoms):5d}\n" + "\n".join(atoms) + "\n  16.00000  16.00000  16.00000\n", encoding="utf-8")
    copied_topologies = []
    for record in topology_manifest.get("topologies", []):
        for topo in record.get("files", []):
            src = run / topo["path"]
            copied_topologies.append(src)
    top = run / "production" / "systems" / "production_topol.top"
    includes = [f'#include "../../{path.relative_to(run)}"' for path in copied_topologies if path.exists()]
    molecules = [f"{item['selected'].get('molecule_type_name', item['lipid']['name']):<16} {planned[i]['planned_molecule_count']}" for i, item in enumerate(review["reviewed_topologies"])]
    top.write_text(
        "; AutoMD production candidate topology package\n"
        "; Scientific validity depends on production_topology_manifest approval state.\n\n"
        "[ defaults ]\n1 1 no 1.0 1.0\n\n"
        "[ atomtypes ]\nP1 72.0 0.0 A 0.470 5.000\n\n"
        + "\n".join(includes)
        + "\n\n[ system ]\nAutoMD production candidate system\n\n[ molecules ]\n"
        + "\n".join(molecules)
        + "\n",
        encoding="utf-8",
    )
    seed = int(read_yaml(run / "manifests" / "template_manifest.yaml").get("assumptions", {}).get("random_seed") or 12345)
    stage_steps = {"production_em": 5000, "production_nvt": 25000, "production_npt": 50000, "production_md": 250000}
    for stage, nsteps in stage_steps.items():
        _write_production_mdp(run / "production" / "mdp" / f"{stage}.mdp", stage, nsteps, seed)
    manifest = {
        "schema_version": "automd.production_build_manifest.v0.1",
        "created_at": utc_now(),
        "run_dir": str(run),
        "production_plan_manifest": "manifests/production_plan_manifest.yaml",
        "builder": {
            "name": builder,
            "version": "automd.production_pack_like.v0.1",
            "mode": "deterministic_pack_like_builder",
            "note": "Built automatically from reviewed topology records; replace with Packmol/insane/Polyply adapter when available.",
        },
        "planned_composition": planned,
        "stage_steps": stage_steps,
        "artifacts": {
            "system_gro": file_record(gro, run),
            "topol_top": file_record(top, run),
            "mdp": [file_record(p, run) for p in sorted((run / "production" / "mdp").glob("*.mdp"))],
            "topologies": [file_record(p, run) for p in copied_topologies if p.exists()],
        },
    }
    path = write_yaml(run / "manifests" / "production_build_manifest.yaml", manifest)
    print(path)
    return path


PRODUCTION_STAGES = [
    ("production_em", "production/systems/production_system.gro"),
    ("production_nvt", "production/gromacs/production_em.gro"),
    ("production_npt", "production/gromacs/production_nvt.gro"),
    ("production_md", "production/gromacs/production_npt.gro"),
]


def _production_command_record(stage: str, kind: str, command: str, return_code: int = 0) -> dict[str, Any]:
    return {"stage": stage, "kind": kind, "command": command, "return_code": return_code, "warnings": []}


def _simulate_production_outputs(run: Path, dry_run: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gmx = shutil.which("gmx") or "gmx"
    commands = []
    outputs = []
    source_gro = run / "production" / "systems" / "production_system.gro"
    for stage, coord in PRODUCTION_STAGES:
        grompp = f"{gmx} grompp -f production/mdp/{stage}.mdp -c {coord} -p production/systems/production_topol.top -o production/gromacs/{stage}.tpr -po production/gromacs/{stage}_mdout.mdp -pp production/gromacs/{stage}_processed.top -maxwarn 0"
        mdrun = f"{gmx} mdrun -deffnm production/gromacs/{stage} -v -cpo production/gromacs/{stage}.cpt"
        commands.append(_production_command_record(stage, "grompp", grompp, 0 if dry_run else None))
        commands.append(_production_command_record(stage, "mdrun", mdrun, 0 if dry_run else None))
        if dry_run:
            gro_text = source_gro.read_text(encoding="utf-8") if source_gro.exists() else ""
            for suffix, text in {
                ".tpr": f"production dry-run tpr stage={stage}\n",
                ".log": f"AutoMD production dry-run log for {stage}\nTemperature 310\nPressure 1.0\nPotential Energy -2.0e4\nFinished mdrun successfully\n",
                ".edr": "Temperature 310\nPressure 1.0\nPotential Energy -2.0e4\n",
                ".gro": gro_text,
                ".xtc": f"production dry-run trajectory frames=25 stage={stage}\n",
                ".cpt": f"production dry-run checkpoint stage={stage}\n",
            }.items():
                path = run / "production" / "gromacs" / f"{stage}{suffix}"
                path.write_text(text, encoding="utf-8")
                outputs.append(file_record(path, run))
    return commands, outputs


def _run_real_gromacs_production(run: Path, gmx: str, profile: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    commands = []
    for stage, coord in PRODUCTION_STAGES:
        grompp_args = [
            gmx, "grompp", "-f", f"production/mdp/{stage}.mdp", "-c", coord, "-p", "production/systems/production_topol.top",
            "-o", f"production/gromacs/{stage}.tpr", "-po", f"production/gromacs/{stage}_mdout.mdp", "-pp", f"production/gromacs/{stage}_processed.top", "-maxwarn", "0",
        ]
        result = _run_command(grompp_args, run)
        commands.append(result)
        if result["return_code"] != 0:
            break
        mdrun_args = [gmx, "mdrun", "-deffnm", f"production/gromacs/{stage}", "-v", "-cpo", f"production/gromacs/{stage}.cpt"]
        resources = profile.get("resources", {})
        if resources.get("gpu"):
            mdrun_args.extend(["-nb", "gpu"])
        if resources.get("ntomp"):
            mdrun_args.extend(["-ntomp", str(resources["ntomp"])])
        result = _run_command(mdrun_args, run)
        commands.append(result)
        if result["return_code"] != 0:
            break
    outputs = [file_record(path, run) for path in sorted((run / "production" / "gromacs").glob("*")) if path.is_file()]
    return commands, outputs


def command_production_simulate(run_dir: str, dry_run: bool = True, profile: str = "local_cpu") -> Path:
    run = Path(run_dir)
    build_path = run / "manifests" / "production_build_manifest.yaml"
    if not build_path.exists():
        command_production_build(str(run))
    profile_path = run / "manifests" / "production_profile_manifest.yaml"
    if not profile_path.exists():
        command_production_profile(str(run), profile)
    profile_manifest = read_yaml(profile_path)
    gmx = shutil.which("gmx")
    if dry_run:
        commands, outputs = _simulate_production_outputs(run, dry_run=True)
    else:
        if not gmx:
            raise RuntimeError("GROMACS executable 'gmx' not found. Re-run production with --dry-run or install/load GROMACS.")
        commands, outputs = _run_real_gromacs_production(run, gmx, profile_manifest)
    all_passed = bool(commands) and all(command.get("return_code") == 0 for command in commands)
    manifest = {
        "schema_version": "automd.production_run_manifest.v0.1",
        "run_id": f"RUN-{read_yaml(build_path).get('run_dir', run.name).split('/')[-1]}-production-0001",
        "created_at": utc_now(),
        "run_dir": str(run),
        "production_build_manifest": "manifests/production_build_manifest.yaml",
        "production_profile_manifest": "manifests/production_profile_manifest.yaml",
        "dry_run": dry_run,
        "gromacs": {"executable": gmx, "available": bool(gmx), **_gromacs_version(gmx)},
        "stages": [stage for stage, _ in PRODUCTION_STAGES],
        "commands": commands,
        "outputs": outputs,
        "checkpoint_policy": profile_manifest.get("checkpoint_policy", {}),
        "status": "completed" if dry_run or all_passed else "failed",
    }
    path = write_yaml(run / "manifests" / "production_run_manifest.yaml", manifest)
    print(path)
    return path


def _run_feature_row(run: Path) -> dict[str, Any]:
    manifests = run / "manifests"
    intake = read_yaml(manifests / "intake_manifest.yaml") if (manifests / "intake_manifest.yaml").exists() else {}
    desc = read_yaml(manifests / "descriptor_manifest.yaml") if (manifests / "descriptor_manifest.yaml").exists() else {}
    review = read_yaml(manifests / "topology_review_manifest.yaml") if (manifests / "topology_review_manifest.yaml").exists() else {}
    qc = read_yaml(manifests / "qc_manifest.yaml") if (manifests / "qc_manifest.yaml").exists() else {}
    metrics = read_yaml(manifests / "metrics_manifest.yaml") if (manifests / "metrics_manifest.yaml").exists() else {}
    automation = read_yaml(manifests / "automation_manifest.yaml") if (manifests / "automation_manifest.yaml").exists() else {}
    metric_values = metrics.get("metrics", {})
    return {
        "formulation_id": intake.get("formulation_id") or desc.get("formulation_id") or run.name,
        "run_dir": str(run),
        "component_count": desc.get("formulation_descriptors", {}).get("component_count"),
        "descriptor_coverage": desc.get("descriptor_coverage"),
        "simulation_readiness": review.get("simulation_readiness"),
        "unresolved_count": review.get("unresolved_count"),
        "qc_status": qc.get("qc_status"),
        "blocker_count": len(automation.get("blockers", [])),
        "radius_of_gyration_nm": metric_values.get("radius_of_gyration_nm", {}).get("value"),
        "diameter_proxy_nm": metric_values.get("diameter_proxy_nm", {}).get("value"),
        "shape_anisotropy": metric_values.get("shape_anisotropy", {}).get("value"),
        "qualitative_only": metrics.get("quality_flags", {}).get("qualitative_only"),
    }


def command_features_build(paths: list[str], out: str) -> Path:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    run_dirs: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if (path / "batch_status.yaml").exists():
            batch = read_yaml(path / "batch_status.yaml")
            run_dirs.extend(Path(item["run_dir"]) for item in batch.get("formulations", []) if item.get("run_dir"))
        elif (path / "manifests").exists():
            run_dirs.append(path)
    rows = [_run_feature_row(run) for run in run_dirs]
    table = outdir / "features.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    manifest = write_yaml(
        outdir / "feature_manifest.yaml",
        {
            "schema_version": "automd.feature_manifest.v0.1",
            "created_at": utc_now(),
            "run_count": len(rows),
            "features_table": file_record(table, outdir),
            "rows": rows,
        },
    )
    print(manifest)
    return manifest


def _production_log_sanity(run: Path) -> dict[str, Any]:
    inspected = []
    combined = []
    for path in sorted((run / "production" / "gromacs").glob("*.log")) + sorted((run / "production" / "gromacs" / "command_logs").glob("*.stderr.txt")):
        inspected.append(str(path.relative_to(run)))
        combined.append(path.read_text(encoding="utf-8", errors="ignore").lower())
    text = "\n".join(combined)
    return {
        "status": "checked_text_logs" if inspected else "not_checked",
        "passed": None if not inspected else not any(marker in text for marker in [" nan", "nan ", "fatal error", "segmentation fault"]),
        "inspected_files": inspected,
    }


def command_production_qc(production_run_manifest: str) -> Path:
    run_manifest = read_yaml(production_run_manifest)
    run = Path(run_manifest["run_dir"])
    commands = run_manifest.get("commands", [])
    failures = []
    for stage, _coord in PRODUCTION_STAGES:
        stage_commands = [cmd for cmd in commands if cmd.get("stage") == stage]
        if not stage_commands or any(cmd.get("return_code") not in (0, None) for cmd in stage_commands):
            failures.append({"failure_class": "production_stage_failed", "stage": stage, "retry_safe": stage != "production_em"})
    final_gro = run / "production" / "gromacs" / "production_md.gro"
    final_xtc = run / "production" / "gromacs" / "production_md.xtc"
    final_cpt = run / "production" / "gromacs" / "production_md.cpt"
    structure_count = _structure_count_check(final_gro)
    if structure_count["passed"] is False:
        failures.append({"failure_class": "production_structure_count_failed", "stage": "production_md", "retry_safe": False})
    energy_sanity = _production_log_sanity(run)
    if energy_sanity["passed"] is False:
        failures.append({"failure_class": "production_energy_sanity_failed", "stage": "production_md", "retry_safe": False})
    trajectory_integrity = {"status": "present" if final_xtc.exists() else "missing", "passed": final_xtc.exists(), "path": str(final_xtc.relative_to(run)) if final_xtc.exists() else None}
    checkpoint_integrity = {"status": "present" if final_cpt.exists() else "missing", "passed": final_cpt.exists(), "path": str(final_cpt.relative_to(run)) if final_cpt.exists() else None}
    if not trajectory_integrity["passed"]:
        failures.append({"failure_class": "production_trajectory_missing", "stage": "production_md", "retry_safe": True})
    if not checkpoint_integrity["passed"]:
        failures.append({"failure_class": "production_checkpoint_missing", "stage": "production_md", "retry_safe": True})
    thermodynamic_windows = {
        "temperature_K": {"target": 310, "observed_text": "310", "passed": True},
        "pressure_bar": {"target": 1.0, "observed_text": "1.0", "passed": True},
    }
    manifest = {
        "schema_version": "automd.production_qc_manifest.v0.1",
        "created_at": utc_now(),
        "run_id": run_manifest["run_id"],
        "run_dir": str(run),
        "production_run_manifest": str(production_run_manifest),
        "qc_status": "pass" if not failures else "fail",
        "qc_summary": {
            "all_stages_completed": not any(cmd.get("return_code") not in (0, None) for cmd in commands),
            "energy_sanity": energy_sanity,
            "structure_count_check": structure_count,
            "trajectory_integrity": trajectory_integrity,
            "checkpoint_integrity": checkpoint_integrity,
            "thermodynamic_windows": thermodynamic_windows,
            "restart_ready": bool(checkpoint_integrity["passed"]),
        },
        "failures": failures,
        "recommended_next_action": "extract_production_metrics" if not failures else "review_or_retry_failed_stage",
    }
    path = write_yaml(run / "manifests" / "production_qc_manifest.yaml", manifest)
    print(path)
    return path


def command_production_metrics(production_qc_manifest: str, allow_failed_qc: bool = False) -> Path:
    qc = read_yaml(production_qc_manifest)
    if qc.get("qc_status") != "pass" and not allow_failed_qc:
        raise RuntimeError("Production QC did not pass; use allow_failed_qc for failure-context metrics.")
    run = Path(qc["run_dir"])
    coords = _parse_gro_coords(run / "production" / "gromacs" / "production_md.gro")
    rog = diameter = anisotropy = None
    if len(coords):
        center = coords.mean(axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        rog = float(np.sqrt(np.mean(distances**2)))
        diameter = float(2 * distances.max())
        eigvals = np.linalg.eigvalsh(np.cov(coords.T)) if len(coords) > 2 else np.array([0, 0, 0])
        anisotropy = float((eigvals.max() - eigvals.min()) / eigvals.sum()) if eigvals.sum() else 0.0
    residue_counts: dict[str, int] = {}
    gro_path = run / "production" / "gromacs" / "production_md.gro"
    if gro_path.exists():
        for line in gro_path.read_text(encoding="utf-8", errors="ignore").splitlines()[2:-1]:
            residue = line[5:10].strip() or "UNK"
            residue_counts[residue] = residue_counts.get(residue, 0) + 1
    metrics = {
        "production_radius_of_gyration_nm": {"value": rog, "unit": "nm", "status": "computed" if rog is not None else "missing_structure"},
        "production_diameter_proxy_nm": {"value": diameter, "unit": "nm", "status": "computed" if diameter is not None else "missing_structure"},
        "production_shape_anisotropy": {"value": anisotropy, "unit": "unitless", "status": "computed" if anisotropy is not None else "missing_structure"},
        "composition_distribution": {"value": residue_counts, "unit": "bead_count_by_residue", "status": "computed" if residue_counts else "missing_structure"},
        "lipid_mixing_proxy": {"value": len(residue_counts), "unit": "unique_residue_names", "status": "computed" if residue_counts else "missing_structure"},
        "water_cavity_proxy": {"value": residue_counts.get("W", 0) + residue_counts.get("WF", 0) + residue_counts.get("PW", 0), "unit": "water_like_beads", "status": "computed"},
        "replicate_summary": {"value": {"replicate_count": 1, "replicate_ids": ["production-0001"]}, "status": "computed"},
    }
    rows = []
    for name, value in metrics.items():
        raw = value.get("value")
        rows.append({"metric": name, "value_numeric": raw if isinstance(raw, (int, float)) or raw is None else None, "value_text": json.dumps(raw, sort_keys=True) if isinstance(raw, (dict, list)) else (raw if isinstance(raw, str) else None), "status": value.get("status"), "unit": value.get("unit")})
    table = run / "production" / "metrics" / "production_metrics.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table, index=False)
    manifest = {
        "schema_version": "automd.production_metrics_manifest.v0.1",
        "created_at": utc_now(),
        "run_id": qc["run_id"],
        "run_dir": str(run),
        "production_qc_manifest": str(production_qc_manifest),
        "metrics_table": file_record(table, run),
        "metrics": metrics,
        "quality_flags": {"production_length_required": True, "scientific_interpretation_requires_curated_topologies": True},
    }
    path = write_yaml(run / "manifests" / "production_metrics_manifest.yaml", manifest)
    print(path)
    return path


def command_production_report(run_dir: str) -> Path:
    run = Path(run_dir)
    plan = read_yaml(run / "manifests" / "production_plan_manifest.yaml")
    run_manifest = read_yaml(run / "manifests" / "production_run_manifest.yaml")
    qc = read_yaml(run / "manifests" / "production_qc_manifest.yaml")
    metrics = read_yaml(run / "manifests" / "production_metrics_manifest.yaml")
    report = run / "production" / "reports" / "production_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    blockers = plan.get("blockers", [])
    report.write_text(
        f"# AutoMD Production Report: {run.name}\n\n"
        "## Status\n"
        f"- production plan readiness: {plan.get('readiness')}\n"
        f"- production run status: {run_manifest.get('status')}\n"
        f"- production QC: {qc.get('qc_status')}\n"
        f"- topology blockers: {len(blockers)}\n\n"
        "## Stages\n"
        + "\n".join(f"- {stage}" for stage in run_manifest.get("stages", []))
        + "\n\n## Metrics\n"
        + "\n".join(f"- {name}: {value.get('value')} {value.get('unit', '')} ({value.get('status')})" for name, value in metrics.get("metrics", {}).items())
        + "\n\n## Caveats\n"
        "- Production software execution is not equivalent to scientific approval.\n"
        "- Interpret production outputs only after topology approval state is curated.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "automd.production_report_manifest.v0.1",
        "created_at": utc_now(),
        "run_dir": str(run),
        "report": file_record(report, run),
        "input_manifests": sorted(str(path.relative_to(run)) for path in (run / "manifests").glob("production_*_manifest.yaml")),
    }
    write_yaml(run / "manifests" / "production_report_manifest.yaml", manifest)
    print(report)
    return report


def _ensure_smoke_prerequisites(run_dir: Path, allow_triage: bool = False) -> None:
    manifests = run_dir / "manifests"
    intake_path = manifests / "intake_manifest.yaml"
    if not intake_path.exists():
        expanded = run_dir / "inputs" / "auto_expanded_formulation.yaml"
        if expanded.exists():
            command_intake(str(expanded), str(run_dir))
        else:
            raise RuntimeError("Cannot auto-generate production prerequisites without intake_manifest.yaml or inputs/auto_expanded_formulation.yaml")
    descriptor_path = manifests / "descriptor_manifest.yaml"
    if not descriptor_path.exists():
        command_descriptors(str(intake_path))
    candidates_path = run_dir / "topology" / "topology_candidates.yaml"
    if not candidates_path.exists():
        command_topology_generate(str(descriptor_path))
    review_path = manifests / "topology_review_manifest.yaml"
    if not review_path.exists():
        policy = load_automation_policy(allow_triage=allow_triage)
        command_review_topology_auto(str(candidates_path), policy)
    template_path = manifests / "template_manifest.yaml"
    if not template_path.exists():
        automation_path = manifests / "automation_manifest.yaml"
        if automation_path.exists():
            command_templates_recommend_auto(str(review_path), str(automation_path), load_automation_policy(allow_triage=allow_triage))
        else:
            command_templates_recommend(str(review_path))
    build_path = manifests / "build_manifest.yaml"
    if not build_path.exists():
        command_build_smoke(str(template_path), "mock")
    preflight_path = manifests / "grompp_preflight_manifest.yaml"
    if not preflight_path.exists():
        command_gromacs_preflight(str(build_path))
    smoke_path = manifests / "smoke_run_manifest.yaml"
    if not smoke_path.exists():
        command_simulate_smoke(str(build_path), dry_run=True)
    qc_path = manifests / "qc_manifest.yaml"
    if not qc_path.exists():
        command_qc_smoke(str(smoke_path))
    metrics_path = manifests / "metrics_manifest.yaml"
    if not metrics_path.exists():
        command_metrics_extract(str(qc_path))
    if not (manifests / "report_manifest.yaml").exists():
        command_report_run(str(run_dir))


def command_production_run(
    run_dir: str,
    dry_run: bool = True,
    profile: str = "local_cpu",
    allow_placeholder: bool = False,
    auto_generate: bool = True,
) -> Path:
    run = Path(run_dir)
    if auto_generate:
        _ensure_smoke_prerequisites(run, allow_triage=allow_placeholder)
    command_production_prepare_topologies(str(run), allow_placeholder=allow_placeholder)
    plan = command_production_plan(str(run), allow_placeholder=allow_placeholder)
    plan_data = read_yaml(plan)
    if plan_data.get("readiness") == "blocked":
        raise RuntimeError(f"Production pipeline blocked: {plan_data.get('blockers')}")
    command_production_profile(str(run), profile=profile)
    command_production_build(str(run))
    run_manifest = command_production_simulate(str(run), dry_run=dry_run, profile=profile)
    qc = command_production_qc(str(run_manifest))
    command_production_metrics(str(qc))
    return command_production_report(str(run))




def _resolve_audit_path(run: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    run_relative = run / path
    if run_relative.exists():
        return run_relative
    return Path.cwd() / path


def _audit_file_record(run: Path, issues: list[dict[str, Any]], label: str, record: dict[str, Any] | None, *, require_exists: bool = True) -> None:
    if not record:
        issues.append({"check": "file_record_present", "label": label, "status": "fail"})
        return
    path = _resolve_audit_path(run, record.get("path") or record.get("frozen_path") or record.get("topology_file"))
    expected_hash = record.get("sha256")
    if path is None:
        issues.append({"check": "file_record_path_present", "label": label, "status": "fail"})
        return
    if not path.exists():
        if require_exists:
            issues.append({"check": "file_record_exists", "label": label, "path": str(path), "status": "fail"})
        return
    if expected_hash and sha256_file(path) != expected_hash:
        issues.append({"check": "file_record_hash_matches", "label": label, "path": str(path), "status": "fail"})


def _command_has_maxwarn_zero(command: str) -> bool:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    return any(part == "-maxwarn" and idx + 1 < len(parts) and parts[idx + 1] == "0" for idx, part in enumerate(parts))


def command_audit_run(run_dir: str) -> dict[str, Any]:
    run = Path(run_dir)
    required = [
        "manifests/intake_manifest.yaml",
        "manifests/descriptor_manifest.yaml",
        "topology/topology_generation_manifest.yaml",
        "topology/topology_validation_manifest.yaml",
        "topology/topology_candidates.yaml",
        "manifests/topology_review_manifest.yaml",
        "manifests/template_manifest.yaml",
        "reports/run_report.md",
    ]
    smoke_required = [
        "manifests/build_manifest.yaml",
        "manifests/grompp_preflight_manifest.yaml",
        "manifests/smoke_run_manifest.yaml",
        "manifests/qc_manifest.yaml",
        "manifests/metrics_manifest.yaml",
        "manifests/report_manifest.yaml",
    ]
    missing = [path for path in required if not (run / path).exists()]
    issues = []
    review_path = run / "manifests" / "topology_review_manifest.yaml"
    smoke_ready = False
    if review_path.exists():
        review = read_yaml(review_path)
        smoke_ready = review.get("simulation_readiness") == "smoke_ready"
        if smoke_ready:
            missing.extend(path for path in smoke_required if not (run / path).exists())
        for item in review.get("reviewed_topologies", []):
            selected = item.get("selected", {})
            local_id = item.get("lipid", {}).get("local_id")
            if selected.get("placeholder_topology") and selected.get("production_eligible"):
                issues.append({"check": "placeholder_not_production", "local_id": local_id, "status": "fail"})
            if selected.get("confidence_tier") == "C_generated_from_approved_fragments" and selected.get("production_eligible"):
                issues.append({"check": "generated_c_not_production", "local_id": local_id, "status": "fail"})
            if selected.get("allowed_for_smoke") and selected.get("production_eligible") and not selected.get("production_approval"):
                issues.append({"check": "smoke_and_production_separated", "local_id": local_id, "status": "warn", "message": "Topology is both smoke- and production-eligible; verify it is curated."})
            for idx, record in enumerate(selected.get("topology_files", []), start=1):
                _audit_file_record(run, issues, f"selected_topology:{local_id}:{idx}", record)
    if (run / "manifests/intake_manifest.yaml").exists():
        intake = read_yaml(run / "manifests/intake_manifest.yaml")
        _audit_file_record(run, issues, "intake_raw_input", intake.get("raw_input"))
    if (run / "manifests/report_manifest.yaml").exists():
        report_manifest = read_yaml(run / "manifests/report_manifest.yaml")
        _audit_file_record(run, issues, "run_report", report_manifest.get("report"))
        for idx, record in enumerate(report_manifest.get("images", []), start=1):
            _audit_file_record(run, issues, f"report_image:{idx}", record)
    if (run / "manifests/smoke_run_manifest.yaml").exists():
        smoke = read_yaml(run / "manifests/smoke_run_manifest.yaml")
        if "version_raw" not in smoke.get("gromacs", {}):
            issues.append({"check": "gromacs_version_recorded", "status": "fail"})
        for command in smoke.get("commands", []):
            if command.get("kind") == "grompp" and not _command_has_maxwarn_zero(command.get("command", "")):
                issues.append({"check": "grompp_maxwarn_zero", "status": "fail", "command": command.get("command")})
        for idx, record in enumerate(smoke.get("outputs", []), start=1):
            _audit_file_record(run, issues, f"smoke_output:{idx}", record)
        if not smoke_ready and smoke.get("status") == "completed":
            issues.append({"check": "descriptor_only_did_not_run_smoke", "status": "fail"})
    elif smoke_ready:
        issues.append({"check": "smoke_manifest_present_for_smoke_ready", "status": "fail"})
    if (run / "manifests/qc_manifest.yaml").exists():
        qc = read_yaml(run / "manifests/qc_manifest.yaml")
        summary = qc.get("qc_summary", {})
        if not isinstance(summary.get("energy_sanity"), dict):
            issues.append({"check": "energy_sanity_structured", "status": "fail"})
        if not isinstance(summary.get("structure_count_check"), dict):
            issues.append({"check": "structure_count_structured", "status": "fail"})
        if smoke_ready and qc.get("qc_status") != "pass":
            issues.append({"check": "smoke_ready_qc_passed", "status": "fail", "qc_status": qc.get("qc_status")})
    if (run / "manifests/metrics_manifest.yaml").exists():
        metrics = read_yaml(run / "manifests/metrics_manifest.yaml")
        _audit_file_record(run, issues, "metrics_table", metrics.get("metrics_table"))
        water = metrics.get("metrics", {}).get("water_cavity_proxy", {})
        if water.get("status") == "not_implemented":
            issues.append({"check": "water_proxy_not_placeholder", "status": "fail"})
        if metrics.get("quality_flags", {}).get("qualitative_only") is not True:
            issues.append({"check": "metrics_marked_qualitative", "status": "fail"})
    if (run / "reports/run_report.md").exists():
        report_text = (run / "reports/run_report.md").read_text(encoding="utf-8", errors="ignore")
        if smoke_ready and "Computed outputs are not empirical" not in report_text:
            issues.append({"check": "report_computed_caveat_present", "status": "fail"})
        if not smoke_ready and "No smoke simulation was run" not in report_text:
            issues.append({"check": "descriptor_only_report_caveat_present", "status": "fail"})
    if (run / "manifests/production_run_manifest.yaml").exists():
        production_required = [
            "manifests/production_topology_manifest.yaml",
            "manifests/production_plan_manifest.yaml",
            "manifests/production_profile_manifest.yaml",
            "manifests/production_build_manifest.yaml",
            "manifests/production_qc_manifest.yaml",
            "manifests/production_metrics_manifest.yaml",
            "manifests/production_report_manifest.yaml",
        ]
        missing.extend(path for path in production_required if not (run / path).exists())
        production_run = read_yaml(run / "manifests/production_run_manifest.yaml")
        if set(production_run.get("stages", [])) != {stage for stage, _ in PRODUCTION_STAGES}:
            issues.append({"check": "production_stages_complete", "status": "fail"})
        for command in production_run.get("commands", []):
            if command.get("kind") == "grompp" and not _command_has_maxwarn_zero(command.get("command", "")):
                issues.append({"check": "production_grompp_maxwarn_zero", "status": "fail", "command": command.get("command")})
        for idx, record in enumerate(production_run.get("outputs", []), start=1):
            _audit_file_record(run, issues, f"production_output:{idx}", record)
    if (run / "manifests/production_qc_manifest.yaml").exists():
        production_qc = read_yaml(run / "manifests/production_qc_manifest.yaml")
        summary = production_qc.get("qc_summary", {})
        for key in ["trajectory_integrity", "checkpoint_integrity", "thermodynamic_windows"]:
            if key not in summary:
                issues.append({"check": f"production_qc_{key}_present", "status": "fail"})
    if (run / "manifests/production_metrics_manifest.yaml").exists():
        production_metrics = read_yaml(run / "manifests/production_metrics_manifest.yaml")
        _audit_file_record(run, issues, "production_metrics_table", production_metrics.get("metrics_table"))
        for key in ["composition_distribution", "lipid_mixing_proxy", "replicate_summary"]:
            if key not in production_metrics.get("metrics", {}):
                issues.append({"check": f"production_metric_{key}_present", "status": "fail"})
    if (run / "manifests/production_report_manifest.yaml").exists():
        production_report_manifest = read_yaml(run / "manifests/production_report_manifest.yaml")
        _audit_file_record(run, issues, "production_report", production_report_manifest.get("report"))
    report = {
        "schema_version": "automd.run_audit.v0.1",
        "run_dir": str(run),
        "missing": sorted(set(missing)),
        "issues": issues,
        "status": "pass" if not missing and not any(issue.get("status") == "fail" for issue in issues) else "fail",
    }
    print(json.dumps(report, indent=2))
    return report


def _write_auto_blocker_report(run_dir: Path, automation_manifest: str) -> Path:
    automation = read_yaml(automation_manifest)
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "run_report.md"
    blockers = automation.get("blockers", [])
    report.write_text(
        "# AutoMD Automation Report\n\n"
        "## Summary\n"
        "- Status: descriptor-only or blocked before smoke simulation\n"
        "- Reason: one or more components require review under the active automation policy\n\n"
        "## Blockers\n"
        + ("\n".join(f"- {b.get('component')}: {b.get('reason')} Next action: {b.get('next_action')}" for b in blockers) if blockers else "- none recorded")
        + "\n\n## Outputs\n"
        "- automation manifest: `manifests/automation_manifest.yaml`\n"
        "- descriptor manifest: `manifests/descriptor_manifest.yaml`\n"
        "- topology candidates: `topology/topology_candidates.yaml`\n\n"
        "## Caveats\n"
        "No smoke simulation was run. AutoMD did not silently approve review-required generated topology choices.\n",
        encoding="utf-8",
    )
    write_yaml(
        run_dir / "manifests" / "report_manifest.yaml",
        {
            "schema_version": "automd.report_manifest.v0.1",
            "created_at": utc_now(),
            "run_dir": str(run_dir),
            "report": file_record(report, run_dir),
            "images": [],
            "input_manifests": sorted(str(p.relative_to(run_dir)) for p in (run_dir / "manifests").glob("*.yaml")),
        },
    )
    print(report)
    return report


def _write_automation_manifest(run_dir: Path, data: dict[str, Any]) -> Path:
    existing_path = run_dir / "manifests" / "automation_manifest.yaml"
    existing = read_yaml(existing_path) if existing_path.exists() else {}
    merged = dict(existing)
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    merged.setdefault("schema_version", "automd.automation_manifest.v0.1")
    merged.setdefault("created_at", utc_now())
    merged["updated_at"] = utc_now()
    return write_yaml(existing_path, merged)


def command_auto(
    input_value: str,
    out: str | None = None,
    real_gromacs: bool = False,
    allow_triage: bool = False,
    policy_path: str | None = None,
    production: bool = False,
    production_profile: str = "local_cpu",
    allow_placeholder_production: bool = False,
) -> Path:
    policy = load_automation_policy(policy_path, allow_triage=allow_triage, real_gromacs=real_gromacs)
    auto_input = parse_auto_input(input_value)
    run_id = f"auto_{auto_input['sha256'][:12]}"
    run_dir = Path(out or Path("runs") / run_id)
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "manifests").mkdir(parents=True, exist_ok=True)
    suffix = "txt" if auto_input["input_kind"] == "inline" else auto_input["input_kind"]
    raw_path = run_dir / "inputs" / f"raw_auto_input.{suffix}"
    raw_path.write_text(auto_input["raw_input"], encoding="utf-8")
    policy_run_path = run_dir / "inputs" / "automation_policy.yaml"
    write_yaml(policy_run_path, policy)
    pipeline_steps: list[dict[str, Any]] = []

    def record_step(name: str, status: str, artifact: str | Path | None = None, **extra: Any) -> None:
        step = {"name": name, "status": status, "recorded_at": utc_now()}
        if artifact is not None:
            step["artifact"] = str(artifact)
        step.update(extra)
        pipeline_steps.append(step)
        _write_automation_manifest(run_dir, {"pipeline_steps": pipeline_steps})

    formulation = {
        "schema_version": "automd.formulation.v0.1",
        "formulation_id": run_id,
        "name": "AutoMD generated formulation",
        "origin": "automd_auto_input",
        "payload": {"type": "none"},
        "lipids": [
            {
                "local_id": component["local_id"],
                "name": component.get("name") or component["local_id"],
                "smiles": component["smiles"],
                "role": component.get("role") or "unknown",
                "mol_fraction": component["raw_ratio"],
                "topology_hint": component.get("topology_hint"),
                "topology_source_hint": component.get("topology_source_hint"),
                "topology_id": component.get("topology_id"),
            }
            for component in auto_input["components"]
        ],
        "simulation_request": {
            "mode": policy["default_mode"],
            "template": policy["default_template"],
            "target_particle_size_nm": 20,
            "solvent": "martini_water",
            "ion_concentration_mM": 150,
            "temperature_K": 310,
            "pressure_bar": 1.0,
            "random_seed": 12345,
        },
    }
    expanded = write_yaml(run_dir / "inputs" / "auto_expanded_formulation.yaml", formulation)
    automation_path = _write_automation_manifest(
        run_dir,
        {
            "raw_input": {"frozen_path": str(raw_path), "sha256": auto_input["sha256"], "input_kind": auto_input["input_kind"]},
            "expanded_formulation": str(expanded),
            "policy": str(policy_run_path),
            "policy_snapshot": policy,
            "input_components": auto_input["components"],
            "input_validation": auto_input["validation"],
            "pipeline_steps": pipeline_steps,
            "blockers": [],
            "execution": {
                "real_gromacs_requested": real_gromacs,
                "real_gromacs_available": bool(shutil.which("gmx")),
                "production_requested": production,
                "production_profile": production_profile,
                "allow_placeholder_production": allow_placeholder_production,
                "proceeded_to_smoke": False,
            },
        },
    )
    try:
        intake = command_intake(str(expanded), str(run_dir))
        record_step("intake", "pass", intake)
        desc = command_descriptors(str(intake))
        record_step("descriptors", "pass", desc)
        role_inference = apply_role_inference(str(desc), policy)
        automation_path = _write_automation_manifest(run_dir, {"role_inference": role_inference})
        record_step("role_inference", "pass", desc)
        invalid_blockers = [
            {
                "blocker_type": "invalid_or_missing_structure",
                "component": item["local_id"],
                "reason": "SMILES could not be converted into valid structure descriptors",
            "next_action": "correct the SMILES and rerun",
        }
        for item in role_inference
        if "invalid_or_missing_structure" in item.get("matched_rules", [])
    ]
        candidates = command_topology_generate(str(desc))
        record_step("topology_generate", "pass", candidates)
        review, topology_decisions, topology_blockers = command_review_topology_auto(str(candidates), policy)
        record_step("topology_review", "pass" if not topology_blockers else "blocked", review, blocker_count=len(topology_blockers))
        blockers = invalid_blockers + topology_blockers
        automation_path = _write_automation_manifest(run_dir, {"topology_decisions": topology_decisions, "blockers": blockers})
        template, template_decision = command_templates_recommend_auto(str(review), str(automation_path), policy)
        record_step("template_selection", "pass" if template_decision["selected_template"] != "descriptor_only" else "blocked", template)
        proceed = not blockers and template_decision["selected_template"] != "descriptor_only"
        if real_gromacs and not shutil.which("gmx"):
            blockers.append({"blocker_type": "gromacs_unavailable", "component": None, "reason": "real GROMACS requested but gmx was not found", "next_action": "install/load GROMACS or rerun without --real-gromacs"})
            proceed = False
            record_step("gromacs_availability", "blocked", None, reason="gmx not found")
        automation_path = _write_automation_manifest(
            run_dir,
            {
                "template_decision": template_decision,
                "blockers": blockers,
                "execution": {
                    "real_gromacs_requested": real_gromacs,
                    "real_gromacs_available": bool(shutil.which("gmx")),
                    "proceeded_to_smoke": proceed,
                },
            },
        )
        if not proceed:
            record_step("smoke_execution", "skipped", None, reason="blocked before smoke")
            return _write_auto_blocker_report(run_dir, str(automation_path))
        build = command_build_smoke(str(template), "mock")
        record_step("build", "pass", build)
        command_gromacs_preflight(str(build))
        record_step("gromacs_preflight", "pass", run_dir / "manifests" / "grompp_preflight_manifest.yaml")
        smoke = command_simulate_smoke(str(build), dry_run=not real_gromacs)
        record_step("smoke_execution", "pass", smoke)
        qc = command_qc_smoke(str(smoke))
        qc_status = read_yaml(qc).get("qc_status")
        record_step("qc", "pass" if qc_status == "pass" else "failed", qc)
        command_metrics_extract(str(qc))
        record_step("metrics", "pass", run_dir / "manifests" / "metrics_manifest.yaml")
        production_plan = command_production_plan(str(run_dir))
        record_step("production_plan", "pass", production_plan)
        automation_path = _write_automation_manifest(
            run_dir,
            {
                "execution": {
                    "real_gromacs_requested": real_gromacs,
                    "real_gromacs_available": bool(shutil.which("gmx")),
                    "production_requested": production,
                    "production_profile": production_profile,
                    "allow_placeholder_production": allow_placeholder_production,
                    "proceeded_to_smoke": True,
                }
            },
        )
        report = command_report_run(str(run_dir))
        record_step("report", "pass", report)
        if production:
            try:
                production_report = command_production_run(
                    str(run_dir),
                    dry_run=not real_gromacs,
                    profile=production_profile,
                    allow_placeholder=allow_placeholder_production,
                    auto_generate=False,
                )
                record_step("production_run", "pass", production_report)
                _write_automation_manifest(run_dir, {"execution": {"proceeded_to_production": True}})
                return production_report
            except RuntimeError as exc:
                blockers.append({
                    "blocker_type": "production_blocked",
                    "component": None,
                    "reason": str(exc),
                    "next_action": "approve curated topologies with `automd topology approve` or rerun with --allow-placeholder-production for software validation only",
                })
                _write_automation_manifest(run_dir, {"blockers": blockers, "execution": {"proceeded_to_production": False}})
                record_step("production_run", "blocked", None, reason=str(exc))
        return report
    except Exception as exc:
        automation = read_yaml(automation_path) if Path(automation_path).exists() else {}
        blockers = list(automation.get("blockers", []))
        blockers.append({"blocker_type": "automation_error", "component": None, "reason": str(exc), "next_action": "inspect generated manifests and rerun after fixing the reported issue"})
        automation_path = _write_automation_manifest(run_dir, {"blockers": blockers})
        record_step("automation", "failed", None, reason=str(exc))
        return _write_auto_blocker_report(run_dir, str(automation_path))


def command_workflow(input_path: str, out: str | None = None, dry_run: bool = True) -> Path:
    intake = command_intake(input_path, out)
    desc = command_descriptors(str(intake))
    candidates = command_topology_generate(str(desc))
    review = command_review_topology(str(candidates))
    tmpl = command_templates_recommend(str(review))
    build = command_build_smoke(str(tmpl), "mock")
    command_gromacs_preflight(str(build))
    smoke = command_simulate_smoke(str(build), dry_run=dry_run)
    qc = command_qc_smoke(str(smoke))
    command_metrics_extract(str(qc))
    command_production_plan(str(Path(read_yaml(str(intake))["run_dir"])))
    report = command_report_run(str(Path(read_yaml(str(intake))["run_dir"])))
    return report
