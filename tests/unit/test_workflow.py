from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from automd import core


def test_complete_mock_workflow(tmp_path):
    out = tmp_path / "test_tiny"
    report = core.command_workflow("tests/fixtures/tiny_formulation.yaml", str(out), dry_run=True)
    audit = core.command_audit_run(str(out))
    assert report.exists()
    assert audit["status"] == "pass"
    assert (out / "manifests" / "intake_manifest.yaml").exists()
    assert (out / "manifests" / "descriptor_manifest.yaml").exists()
    assert (out / "manifests" / "topology_review_manifest.yaml").exists()
    assert (out / "manifests" / "template_manifest.yaml").exists()
    assert (out / "manifests" / "build_manifest.yaml").exists()
    assert (out / "manifests" / "smoke_run_manifest.yaml").exists()
    assert (out / "manifests" / "qc_manifest.yaml").exists()
    assert (out / "manifests" / "metrics_manifest.yaml").exists()
    assert (out / "manifests" / "report_manifest.yaml").exists()
    assert (out / "images" / "smoke_snapshot.png").exists()
    assert "AutoMD Run Card" in report.read_text()


def test_ratio_normalization_and_raw_hash(tmp_path):
    manifest_path = core.command_intake("tests/fixtures/tiny_formulation.yaml", str(tmp_path / "tiny"))
    data = yaml.safe_load(Path(manifest_path).read_text())
    assert data["ratio_policy"]["interpreted_as"] == "mol_percent"
    assert data["ratio_policy"]["raw_total"] == 100.0
    assert data["raw_input"]["sha256"]


def test_auto_inline_parser_normalizes_and_hashes():
    parsed = core.parse_auto_input("CCO:25,CCCC:75")
    assert parsed["input_kind"] == "inline"
    assert parsed["sha256"]
    assert parsed["ratio_policy"]["raw_total"] == 100.0
    assert [component["local_id"] for component in parsed["components"]] == ["component_001", "component_002"]
    assert [component["normalized_mol_fraction"] for component in parsed["components"]] == [0.25, 0.75]


def test_auto_csv_parser(tmp_path):
    csv_path = tmp_path / "components.csv"
    csv_path.write_text("local_id,name,smiles,ratio,role\nethanol,Ethanol,CCO,40,small_molecule\ndodecane,Dodecane,CCCCCCCCCCCC,60,additive\n", encoding="utf-8")
    parsed = core.parse_auto_input(str(csv_path))
    assert parsed["input_kind"] == "csv"
    assert parsed["source"] == str(csv_path)
    assert parsed["components"][0]["smiles"] == "CCO"
    assert parsed["components"][0]["name"] == "Ethanol"
    assert parsed["components"][0]["role"] == "small_molecule"
    assert parsed["components"][1]["raw_ratio"] == 60.0


def test_auto_inline_parser_accepts_semicolon_and_newline():
    parsed = core.parse_auto_input("CCO:25;CCCC:25\nCCCCCCCC:50")
    assert parsed["validation"]["component_count"] == 3
    assert [component["normalized_mol_fraction"] for component in parsed["components"]] == [0.25, 0.25, 0.5]


def test_descriptors_parquet_loadable(tmp_path):
    intake = core.command_intake("tests/fixtures/tiny_formulation.yaml", str(tmp_path / "tiny"))
    desc = core.command_descriptors(str(intake))
    data = yaml.safe_load(Path(desc).read_text())
    table = Path(data["run_dir"]) / data["descriptor_table"]["path"]
    frame = pd.read_parquet(table)
    assert set(frame["descriptor_status"]) == {"computed_from_structure"}


def test_auto_expands_minimal_input_and_freezes_raw_input(tmp_path):
    out = tmp_path / "auto_blocked"
    report = core.command_auto("CCO:100", str(out))
    automation = yaml.safe_load((out / "manifests" / "automation_manifest.yaml").read_text())
    expanded = yaml.safe_load((out / "inputs" / "auto_expanded_formulation.yaml").read_text())
    assert report == out / "reports" / "run_report.md"
    assert (out / "inputs" / "raw_auto_input.txt").read_text(encoding="utf-8") == "CCO:100"
    assert expanded["lipids"][0]["smiles"] == "CCO"
    assert expanded["lipids"][0]["role"] == "unknown"
    assert automation["raw_input"]["sha256"] == core.parse_auto_input("CCO:100")["sha256"]


def test_auto_preserves_input_metadata_pipeline_trace_and_seed(tmp_path):
    csv_path = tmp_path / "auto_metadata.csv"
    csv_path.write_text(
        "local_id,name,smiles,ratio,role\n"
        "amine_lipid,Amine lipid,CCCCCCCCCCCCN(CCCCCCCC)CCCCCCCC,100,ionizable_lipid\n",
        encoding="utf-8",
    )
    out = tmp_path / "auto_metadata"
    report = core.command_auto(str(csv_path), str(out))
    expanded = yaml.safe_load((out / "inputs" / "auto_expanded_formulation.yaml").read_text())
    automation = yaml.safe_load((out / "manifests" / "automation_manifest.yaml").read_text())
    build = yaml.safe_load((out / "manifests" / "build_manifest.yaml").read_text())
    mdp = (out / "mdp" / "smoke_nvt.mdp").read_text()
    production = yaml.safe_load((out / "manifests" / "production_plan_manifest.yaml").read_text())
    assert report.exists()
    assert expanded["lipids"][0]["name"] == "Amine lipid"
    assert expanded["lipids"][0]["role"] == "ionizable_lipid"
    assert [step["name"] for step in automation["pipeline_steps"]][-1] == "report"
    assert build["reproducibility"]["random_seed"] == 12345
    assert "gen-seed = 12345" in mdp
    assert production["readiness"] == "blocked"
    assert any(blocker["blocker_type"] == "placeholder_topology" for blocker in production["blockers"])


def test_auto_role_inference_recognizes_sterol_peg_and_small_molecule():
    policy = core.load_automation_policy()
    sterol = core.infer_component_role(
        {
            "local_id": "component_001",
            "descriptor_status": "computed_from_structure",
            "canonical_smiles": "CC(C)CCCC(C)C1CCC2C3CCC4=CC(O)CCC4(C)C3CCC12C",
            "ring_count": 4,
            "exact_mw": 386.0,
            "heteroatom_count": 1,
            "rotatable_bonds": 5,
        },
        policy,
    )
    peg = core.infer_component_role(
        {
            "local_id": "component_002",
            "descriptor_status": "computed_from_structure",
            "canonical_smiles": "CCCCCCCCOCCOCCOCCOCC",
            "ring_count": 0,
            "exact_mw": 306.0,
            "heteroatom_count": 5,
            "rotatable_bonds": 18,
        },
        policy,
    )
    small = core.infer_component_role(
        {
            "local_id": "component_003",
            "descriptor_status": "computed_from_structure",
            "canonical_smiles": "CCO",
            "ring_count": 0,
            "exact_mw": 46.0,
            "heteroatom_count": 1,
            "rotatable_bonds": 0,
        },
        policy,
    )
    assert sterol["inferred_role"] == "sterol"
    assert peg["inferred_role"] == "peg_lipid"
    assert small["inferred_role"] == "small_molecule"


def test_unresolved_routes_descriptor_only(tmp_path):
    intake = core.command_intake("examples/demo_unresolved_topology.yaml", str(tmp_path / "unresolved"))
    desc = core.command_descriptors(str(intake))
    candidates = core.command_topology_resolve(str(desc))
    review = core.command_review_topology(str(candidates))
    tmpl = core.command_templates_recommend(str(review))
    data = yaml.safe_load(Path(tmpl).read_text())
    assert data["selected_template"]["template_id"] == "descriptor_only"


def test_topology_generate_family_template_and_validation(tmp_path):
    intake = core.command_intake("examples/demo_unresolved_topology.yaml", str(tmp_path / "generated_family"))
    desc = core.command_descriptors(str(intake))
    candidates = core.command_topology_generate(str(desc))
    run = tmp_path / "generated_family"
    generation = yaml.safe_load((run / "topology" / "topology_generation_manifest.yaml").read_text())
    validation = yaml.safe_load((run / "topology" / "topology_validation_manifest.yaml").read_text())
    packet = yaml.safe_load(Path(candidates).read_text())["review_packets"][0]
    selected = generation["generations"][0]["selected"]
    assert selected["backend"] == "FragmentTemplateBackend"
    assert selected["confidence_tier"] == "C_generated_from_approved_fragments"
    assert selected["smoke_eligible"] is True
    assert selected["production_eligible"] is False
    assert selected["production_review_required"] is True
    assert validation["validations"][0]["validation_status"] == "pass"
    assert validation["validations"][0]["production_eligible"] is False
    assert packet["auto_selected_option_id"] == 1
    manual_validation = core.command_topology_validate([selected["topology_files"][0]["path"]], str(tmp_path / "manual_validation.yaml"))
    manual = yaml.safe_load(Path(manual_validation).read_text())
    assert manual["validations"][0]["gates"]["confidence_tier_stored"] is True


def test_topology_generate_smallmol_requires_review(tmp_path):
    formulation = tmp_path / "smallmol.yaml"
    formulation.write_text(
        yaml.safe_dump({
            "schema_version": "automd.formulation.v0.1",
            "formulation_id": "smallmol",
            "payload": {"type": "none"},
            "lipids": [{
                "local_id": "lipid_001",
                "name": "small additive",
                "smiles": "CCO",
                "role": "custom_role",
                "mol_fraction": 1.0,
            }],
            "simulation_request": {"mode": "smoke", "template": "auto"},
        }),
        encoding="utf-8",
    )
    intake = core.command_intake(str(formulation), str(tmp_path / "smallmol"))
    desc = core.command_descriptors(str(intake))
    candidates = core.command_topology_generate(str(desc))
    packet = yaml.safe_load(Path(candidates).read_text())["review_packets"][0]
    selected = packet["candidates"][0]
    assert selected["backend"] == "PolyplyBackend"
    assert selected["confidence_tier"] == "D_generated_smallmol_backend"
    assert selected["allowed_for_production"] is False
    assert packet["auto_selected_option_id"] is None
    review = core.command_review_topology(str(candidates))
    reviewed = yaml.safe_load(Path(review).read_text())
    assert reviewed["simulation_readiness"] == "descriptor_only"


def _unknown_fallback_policy(tmp_path):
    policy = tmp_path / "unknown_fallback_policy.yaml"
    data = core.load_automation_policy()
    data["role_inference"]["low_confidence_role_fallback"] = "unknown"
    policy.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(policy)


def test_auto_review_accepts_generated_family_topology(tmp_path):
    intake = core.command_intake("examples/demo_unresolved_topology.yaml", str(tmp_path / "auto_review"))
    desc = core.command_descriptors(str(intake))
    candidates = core.command_topology_generate(str(desc))
    review, decisions, blockers = core.command_review_topology_auto(str(candidates), core.load_automation_policy())
    reviewed = yaml.safe_load(Path(review).read_text())
    assert blockers == []
    assert decisions[0]["confidence_tier"] == "C_generated_from_approved_fragments"
    assert decisions[0]["auto_accepted_for_smoke"] is True
    assert decisions[0]["production_eligible"] is False
    assert decisions[0]["production_requires_review"] is True
    assert reviewed["review_mode"] == "automated"
    assert reviewed["simulation_readiness"] == "smoke_ready"


def test_auto_d_tier_blocks_by_default(tmp_path):
    out = tmp_path / "auto_smallmol_blocked"
    core.command_auto("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC:100", str(out), policy_path=_unknown_fallback_policy(tmp_path))
    automation = yaml.safe_load((out / "manifests" / "automation_manifest.yaml").read_text())
    review = yaml.safe_load((out / "manifests" / "topology_review_manifest.yaml").read_text())
    template = yaml.safe_load((out / "manifests" / "template_manifest.yaml").read_text())
    assert automation["execution"]["proceeded_to_smoke"] is False
    assert automation["blockers"][0]["blocker_type"] == "topology_review_required"
    assert review["simulation_readiness"] == "descriptor_only"
    assert template["selected_template"]["template_id"] == "descriptor_only"
    assert not (out / "manifests" / "smoke_run_manifest.yaml").exists()


def test_auto_d_tier_allow_triage_proceeds_to_dry_run(tmp_path):
    out = tmp_path / "auto_smallmol_triage"
    report = core.command_auto("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC:100", str(out), allow_triage=True, policy_path=_unknown_fallback_policy(tmp_path))
    automation = yaml.safe_load((out / "manifests" / "automation_manifest.yaml").read_text())
    smoke = yaml.safe_load((out / "manifests" / "smoke_run_manifest.yaml").read_text())
    qc = yaml.safe_load((out / "manifests" / "qc_manifest.yaml").read_text())
    metrics = yaml.safe_load((out / "manifests" / "metrics_manifest.yaml").read_text())
    assert report.exists()
    assert automation["blockers"] == []
    assert automation["execution"]["proceeded_to_smoke"] is True
    assert smoke["dry_run"] is True
    assert "version_raw" in smoke["gromacs"]
    assert qc["qc_status"] == "pass"
    assert qc["qc_summary"]["energy_sanity"]["status"] == "checked_text_logs"
    assert qc["qc_summary"]["structure_count_check"]["status"] == "checked_structure_atom_count"
    assert qc["qc_summary"]["expected_molecule_counts_present"] is None
    assert metrics["metrics"]["water_cavity_proxy"]["status"] != "not_implemented"


def test_run_audit_detects_missing_smoke_artifact(tmp_path):
    out = tmp_path / "audit_missing"
    core.command_workflow("tests/fixtures/tiny_formulation.yaml", str(out), dry_run=True)
    (out / "manifests" / "metrics_manifest.yaml").unlink()
    audit = core.command_audit_run(str(out))
    assert audit["status"] == "fail"
    assert "manifests/metrics_manifest.yaml" in audit["missing"]


def test_run_audit_detects_hash_mismatch(tmp_path):
    out = tmp_path / "audit_hash_mismatch"
    core.command_workflow("tests/fixtures/tiny_formulation.yaml", str(out), dry_run=True)
    (out / "reports" / "run_report.md").write_text("tampered report\n", encoding="utf-8")
    audit = core.command_audit_run(str(out))
    assert audit["status"] == "fail"
    assert any(issue["check"] == "file_record_hash_matches" and issue["label"] == "run_report" for issue in audit["issues"])


def test_run_audit_requires_grompp_maxwarn_zero(tmp_path):
    out = tmp_path / "audit_maxwarn"
    core.command_workflow("tests/fixtures/tiny_formulation.yaml", str(out), dry_run=True)
    smoke_path = out / "manifests" / "smoke_run_manifest.yaml"
    smoke = yaml.safe_load(smoke_path.read_text())
    smoke["commands"][0]["command"] = smoke["commands"][0]["command"].replace("-maxwarn 0", "-maxwarn 1")
    smoke_path.write_text(yaml.safe_dump(smoke, sort_keys=False), encoding="utf-8")
    audit = core.command_audit_run(str(out))
    assert audit["status"] == "fail"
    assert any(issue["check"] == "grompp_maxwarn_zero" for issue in audit["issues"])


def test_auto_invalid_smiles_routes_to_manual_review_report(tmp_path):
    out = tmp_path / "auto_invalid"
    report = core.command_auto("notasmiles:100", str(out))
    automation = yaml.safe_load((out / "manifests" / "automation_manifest.yaml").read_text())
    generation = yaml.safe_load((out / "topology" / "topology_generation_manifest.yaml").read_text())
    selected = generation["generations"][0]["selected"]
    assert report.exists()
    assert any(blocker["blocker_type"] == "invalid_or_missing_structure" for blocker in automation["blockers"])
    assert selected["confidence_tier"] == "E_manual_review_required"
    assert automation["execution"]["proceeded_to_smoke"] is False


def test_auto_selects_lnp_template_for_recognized_role_pattern(tmp_path):
    out = tmp_path / "auto_lnp_pattern"
    core.command_auto(
        "CCCCCCCCCCCCN(CCCCCCCC)CCCCCCCC:45,"
        "CC(C)CCCC(C)C1CCC2C3CCC4=CC(O)CCC4(C)C3CCC12C:40,"
        "CCCCCCCCOP(=O)(O)OCCN:15",
        str(out),
    )
    template = yaml.safe_load((out / "manifests" / "template_manifest.yaml").read_text())
    automation = yaml.safe_load((out / "manifests" / "automation_manifest.yaml").read_text())
    assert automation["execution"]["proceeded_to_smoke"] is True
    assert template["selected_template"]["template_id"] == "lnp_smoke_self_assembly"


def test_external_topology_generator_backend(tmp_path, monkeypatch):
    formulation = tmp_path / "external.yaml"
    formulation.write_text(
        yaml.safe_dump({
            "schema_version": "automd.formulation.v0.1",
            "formulation_id": "external",
            "payload": {"type": "none"},
            "lipids": [{
                "local_id": "lipid_001",
                "name": "external candidate",
                "role": "custom_role",
                "mol_fraction": 1.0,
            }],
            "simulation_request": {"mode": "smoke", "template": "auto"},
        }),
        encoding="utf-8",
    )
    script = tmp_path / "external_generator.py"
    script.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "out=Path(os.environ['AUTOMD_TOPOLOGY_OUTPUT_DIR'])/'lipid_001.external.itp'\n"
        "out.write_text('[ moleculetype ]\\nEXTLIP 1\\n\\n[ atoms ]\\n1 P1 1 EXTLIP B1 1 0.0 72.0\\n')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AUTOMD_TOPOLOGY_GENERATOR", f"{shutil.which('python') or 'python'} {script}")
    intake = core.command_intake(str(formulation), str(tmp_path / "external"))
    desc = core.command_descriptors(str(intake))
    candidates = core.command_topology_generate(str(desc))
    selected = yaml.safe_load(Path(candidates).read_text())["review_packets"][0]["candidates"][0]
    assert selected["backend"] == "ExternalCommandBackend"
    assert selected["confidence_tier"] == "D_generated_smallmol_backend"
    assert selected["review_required"] is True
    assert selected["production_eligible"] is False


def test_curated_placeholder_topologies_are_smoke_only(tmp_path):
    out = tmp_path / "placeholder_semantics"
    report = core.command_workflow("examples/demo_lnp_001.yaml", str(out), dry_run=True)
    review = yaml.safe_load((out / "manifests" / "topology_review_manifest.yaml").read_text())
    assert report.exists()
    assert review["simulation_readiness"] == "smoke_ready"
    for item in review["reviewed_topologies"]:
        selected = item["selected"]
        assert selected["smoke_eligible"] is True
        assert selected["production_eligible"] is False
        assert selected["production_review_required"] is True
        assert selected["placeholder_topology"] is True


def test_rdkit_unavailable_descriptor_status(monkeypatch):
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "rdkit" or name.startswith("rdkit."):
            raise ImportError("simulated missing rdkit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    result = core._rdkit_descriptors("CCO")
    assert result["descriptor_status"] == "failed_rdkit_unavailable"


def test_env_doctor_reports_missing_gromacs(monkeypatch):
    real_which = shutil.which
    monkeypatch.setattr(core.shutil, "which", lambda name: None if name == "gmx" else real_which(name))
    report = core.command_env_doctor(json_mode=True)
    assert report["gromacs"]["available"] is False
    assert "warning" in report["gromacs"]


def test_cli_help_subprocess():
    result = subprocess.run([sys.executable, "-m", "automd.cli", "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "workflow" in result.stdout
    assert "auto" in result.stdout
    assert "production" in result.stdout
    assert "features" in result.stdout


def test_cli_audit_run_subprocess(tmp_path):
    out = tmp_path / "cli_audit"
    core.command_workflow("tests/fixtures/tiny_formulation.yaml", str(out), dry_run=True)
    result = subprocess.run([sys.executable, "-m", "automd.cli", "audit", "run", str(out)], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert '"status": "pass"' in result.stdout


def test_batch_plan_and_summary(tmp_path):
    plan = core.command_batch_plan("tests/fixtures/batch_formulations.csv", str(tmp_path / "batch"))
    status = core.command_batch_smoke(str(plan), dry_run=True)
    report = core.command_batch_summarize(str(Path(status).parent))
    assert report.exists()
    assert "completed" in report.read_text()


def test_batch_auto_input_priority_features_and_summary(tmp_path):
    batch_csv = tmp_path / "batch_auto.csv"
    batch_csv.write_text(
        "formulation_id,auto_input\n"
        "\"auto_one\",\"CCCCCCCCCCCCN(CCCCCCCC)CCCCCCCC:100\"\n",
        encoding="utf-8",
    )
    batch_dir = tmp_path / "batch_auto"
    plan = core.command_batch_plan(str(batch_csv), str(batch_dir))
    planned = yaml.safe_load(Path(plan).read_text())
    assert planned["execution"]["supports_auto_input"] is True
    assert planned["formulations"][0]["mode"] == "auto"
    status = core.command_batch_smoke(str(plan), dry_run=True)
    status_data = yaml.safe_load(Path(status).read_text())
    run_dir = Path(status_data["formulations"][0]["run_dir"])
    assert status_data["formulations"][0]["audit_status"] == "pass"
    priority = core.command_prioritize([str(run_dir / "manifests" / "descriptor_manifest.yaml")], str(tmp_path / "priority"))
    priority_data = yaml.safe_load(Path(priority).read_text())
    assert priority_data["schema_version"] == "automd.priority_manifest.v0.2"
    assert priority_data["rows"][0]["recommended_next_action"] == "production_plan"
    features = core.command_features_build([str(batch_dir)], str(tmp_path / "features"))
    feature_data = yaml.safe_load(Path(features).read_text())
    assert feature_data["run_count"] == 1
    assert feature_data["rows"][0]["qc_status"] == "pass"
    summary = core.command_batch_summarize(str(batch_dir))
    text = Path(summary).read_text()
    assert "Readiness Counts" in text
    assert "total recorded blockers" in text


def test_production_run_autogenerates_missing_prerequisites_and_contract(tmp_path):
    run = tmp_path / "production_auto_generate"
    core.command_intake("tests/fixtures/tiny_formulation.yaml", str(run))
    report = core.command_production_run(str(run), dry_run=True, profile="local_cpu", allow_placeholder=True)
    assert report == run / "production" / "reports" / "production_report.md"
    expected_manifests = [
        "descriptor_manifest.yaml",
        "topology_review_manifest.yaml",
        "qc_manifest.yaml",
        "metrics_manifest.yaml",
        "production_topology_manifest.yaml",
        "production_plan_manifest.yaml",
        "production_profile_manifest.yaml",
        "production_build_manifest.yaml",
        "production_run_manifest.yaml",
        "production_qc_manifest.yaml",
        "production_metrics_manifest.yaml",
        "production_report_manifest.yaml",
    ]
    for name in expected_manifests:
        assert (run / "manifests" / name).exists()
    plan = yaml.safe_load((run / "manifests" / "production_plan_manifest.yaml").read_text())
    topology = yaml.safe_load((run / "manifests" / "production_topology_manifest.yaml").read_text())
    build = yaml.safe_load((run / "manifests" / "production_build_manifest.yaml").read_text())
    production_run = yaml.safe_load((run / "manifests" / "production_run_manifest.yaml").read_text())
    production_qc = yaml.safe_load((run / "manifests" / "production_qc_manifest.yaml").read_text())
    production_metrics = yaml.safe_load((run / "manifests" / "production_metrics_manifest.yaml").read_text())
    profile = yaml.safe_load((run / "manifests" / "production_profile_manifest.yaml").read_text())
    audit = core.command_audit_run(str(run))
    assert plan["readiness"] == "production_runnable_software_validation"
    assert topology["allow_placeholder_for_software_validation"] is True
    assert build["stage_steps"]["production_md"] > build["stage_steps"]["production_npt"]
    assert production_run["stages"] == ["production_em", "production_nvt", "production_npt", "production_md"]
    assert production_qc["qc_summary"]["trajectory_integrity"]["passed"] is True
    assert production_qc["qc_summary"]["checkpoint_integrity"]["passed"] is True
    assert "composition_distribution" in production_metrics["metrics"]
    assert "replicate_summary" in production_metrics["metrics"]
    assert profile["checkpoint_policy"]["enabled"] is True
    assert audit["status"] == "pass"


def test_production_run_blocks_placeholder_without_explicit_allowance(tmp_path):
    run = tmp_path / "production_blocked"
    core.command_workflow("tests/fixtures/tiny_formulation.yaml", str(run), dry_run=True)
    with pytest.raises(RuntimeError, match="Production pipeline blocked"):
        core.command_production_run(str(run), dry_run=True, allow_placeholder=False)


def test_custom_script_builder_requires_real_outputs(tmp_path):
    run = tmp_path / "custom"
    (run / "manifests").mkdir(parents=True)
    (run / "systems").mkdir()
    script = run / "builder.py"
    script.write_text(
        "from pathlib import Path\n"
        "Path('systems/system.gro').write_text('custom\\n    0\\n   1.0 1.0 1.0\\n')\n"
        "Path('systems/topol.top').write_text('[ system ]\\ncustom\\n\\n[ molecules ]\\n')\n",
        encoding="utf-8",
    )
    intake = {
        "schema_version": "automd.intake_manifest.v0.1",
        "formulation_id": "custom",
        "run_dir": str(run),
        "lipids": [],
    }
    review = {"schema_version": "automd.topology_review_manifest.v0.1", "reviewed_topologies": []}
    template = {
        "schema_version": "automd.template_manifest.v0.1",
        "run_dir": str(run),
        "formulation_id": "custom",
        "topology_review_manifest": str(run / "manifests" / "topology_review_manifest.yaml"),
        "custom_builder": {"command": f"{shutil.which('python') or 'python'} builder.py"},
    }
    (run / "manifests" / "intake_manifest.yaml").write_text(yaml.safe_dump(intake), encoding="utf-8")
    (run / "manifests" / "topology_review_manifest.yaml").write_text(yaml.safe_dump(review), encoding="utf-8")
    template_path = run / "manifests" / "template_manifest.yaml"
    template_path.write_text(yaml.safe_dump(template), encoding="utf-8")
    build = core.command_build_smoke(str(template_path), builder="custom_script")
    data = yaml.safe_load(Path(build).read_text())
    assert data["builder"]["name"] == "custom_script"
    assert data["builder"]["result"]["return_code"] == 0


@pytest.mark.skipif(not shutil.which("gmx"), reason="GROMACS not installed")
def test_real_gromacs_smoke_workflow(tmp_path):
    out = tmp_path / "real_gmx"
    report = core.command_workflow("tests/fixtures/tiny_formulation.yaml", str(out), dry_run=False)
    smoke = yaml.safe_load((out / "manifests" / "smoke_run_manifest.yaml").read_text())
    qc = yaml.safe_load((out / "manifests" / "qc_manifest.yaml").read_text())
    assert report.exists()
    assert smoke["dry_run"] is False
    assert smoke["status"] == "completed"
    assert all(command["return_code"] == 0 for command in smoke["commands"])
    assert qc["qc_status"] == "pass"
    assert (out / "gromacs" / "smoke_npt.xtc").exists()


@pytest.mark.skipif(not shutil.which("gmx"), reason="GROMACS not installed")
def test_generated_family_topology_reaches_real_gromacs_smoke(tmp_path):
    out = tmp_path / "generated_family_real_gmx"
    report = core.command_workflow("examples/demo_unresolved_topology.yaml", str(out), dry_run=False)
    generation = yaml.safe_load((out / "topology" / "topology_generation_manifest.yaml").read_text())
    smoke = yaml.safe_load((out / "manifests" / "smoke_run_manifest.yaml").read_text())
    qc = yaml.safe_load((out / "manifests" / "qc_manifest.yaml").read_text())
    assert report.exists()
    assert generation["generations"][0]["selected"]["confidence_tier"] == "C_generated_from_approved_fragments"
    assert smoke["status"] == "completed"
    assert all(command["return_code"] == 0 for command in smoke["commands"])
    assert qc["qc_status"] == "pass"


@pytest.mark.skipif(not shutil.which("gmx"), reason="GROMACS not installed")
def test_auto_real_gromacs_smoke(tmp_path):
    out = tmp_path / "auto_real_gmx"
    report = core.command_auto("CCCCCCCCCCCCN(CCCCCCCC)CCCCCCCC:100", str(out), real_gromacs=True)
    smoke = yaml.safe_load((out / "manifests" / "smoke_run_manifest.yaml").read_text())
    qc = yaml.safe_load((out / "manifests" / "qc_manifest.yaml").read_text())
    automation = yaml.safe_load((out / "manifests" / "automation_manifest.yaml").read_text())
    assert report.exists()
    assert automation["execution"]["proceeded_to_smoke"] is True
    assert smoke["dry_run"] is False
    assert smoke["status"] == "completed"
    assert qc["qc_status"] == "pass"
