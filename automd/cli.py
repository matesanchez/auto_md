from __future__ import annotations

import argparse
import sys

from . import __version__
from . import core


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        raise SystemExit(f"automd: error: {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = Parser(prog="automd", description="AutoMD: mostly automated Martini 3 + GROMACS simulation preparation")
    parser.add_argument("--version", action="version", version=f"automd {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    env = sub.add_parser("env")
    env_sub = env.add_subparsers(dest="env_cmd")
    env_doctor = env_sub.add_parser("doctor")
    env_doctor.add_argument("--json", action="store_true")

    sources = sub.add_parser("sources")
    sources_sub = sources.add_subparsers(dest="sources_cmd")
    sources_sub.add_parser("list")
    fetch = sources_sub.add_parser("fetch")
    fetch.add_argument("--all", action="store_true")

    auto = sub.add_parser("auto", help="Run from minimal SMILES:ratio input to final report")
    auto.add_argument("input")
    auto.add_argument("--out")
    auto.add_argument("--real-gromacs", action="store_true")
    auto.add_argument("--allow-triage", action="store_true")
    auto.add_argument("--policy")

    intake = sub.add_parser("intake")
    intake.add_argument("input")
    intake.add_argument("--out")

    descriptors = sub.add_parser("descriptors")
    desc_sub = descriptors.add_subparsers(dest="desc_cmd")
    desc_run = desc_sub.add_parser("run")
    desc_run.add_argument("intake_manifest")

    topology = sub.add_parser("topology")
    topo_sub = topology.add_subparsers(dest="topo_cmd")
    topo_index = topo_sub.add_parser("index")
    topo_index.add_argument("--sources")
    topo_index.add_argument("--out", default="topology_library/topology_registry.yaml")
    topo_resolve = topo_sub.add_parser("resolve")
    topo_resolve.add_argument("descriptor_manifest")
    topo_generate = topo_sub.add_parser("generate")
    topo_generate.add_argument("descriptor_manifest")
    topo_validate = topo_sub.add_parser("validate")
    topo_validate.add_argument("topology_files", nargs="+")
    topo_validate.add_argument("--out")

    review = sub.add_parser("review")
    review_sub = review.add_subparsers(dest="review_cmd")
    review_topology = review_sub.add_parser("topology")
    review_topology.add_argument("topology_candidates")
    review_topology.add_argument("--answers")

    templates = sub.add_parser("templates")
    tmpl_sub = templates.add_subparsers(dest="templates_cmd")
    tmpl_sub.add_parser("list")
    tmpl_rec = tmpl_sub.add_parser("recommend")
    tmpl_rec.add_argument("topology_review_manifest")

    build = sub.add_parser("build")
    build_sub = build.add_subparsers(dest="build_cmd")
    build_smoke = build_sub.add_parser("smoke")
    build_smoke.add_argument("template_manifest")
    build_smoke.add_argument("--builder", default="mock", choices=["mock", "custom_script"])

    gromacs = sub.add_parser("gromacs")
    gmx_sub = gromacs.add_subparsers(dest="gmx_cmd")
    preflight = gmx_sub.add_parser("preflight")
    preflight.add_argument("build_manifest")

    simulate = sub.add_parser("simulate")
    sim_sub = simulate.add_subparsers(dest="sim_cmd")
    smoke = sim_sub.add_parser("smoke")
    smoke.add_argument("build_manifest")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--gmx-extra", default="")
    smoke.add_argument("--hpc-profile")

    qc = sub.add_parser("qc")
    qc_sub = qc.add_subparsers(dest="qc_cmd")
    qc_smoke = qc_sub.add_parser("smoke")
    qc_smoke.add_argument("smoke_run_manifest")

    metrics = sub.add_parser("metrics")
    metrics_sub = metrics.add_subparsers(dest="metrics_cmd")
    metrics_extract = metrics_sub.add_parser("extract")
    metrics_extract.add_argument("qc_manifest")
    metrics_extract.add_argument("--allow-failed-qc", action="store_true")

    prioritize = sub.add_parser("prioritize")
    prioritize.add_argument("manifests", nargs="+")
    prioritize.add_argument("--out", required=True)

    production = sub.add_parser("production")
    production_sub = production.add_subparsers(dest="production_cmd")
    production_plan = production_sub.add_parser("plan")
    production_plan.add_argument("run_dir")
    production_plan.add_argument("--out")
    production_plan.add_argument("--allow-placeholder", action="store_true")
    production_run = production_sub.add_parser("run")
    production_run.add_argument("run_dir")
    production_run.add_argument("--dry-run", action="store_true")
    production_run.add_argument("--real-gromacs", action="store_true")
    production_run.add_argument("--profile", default="local_cpu")
    production_run.add_argument("--allow-placeholder", action="store_true")
    production_run.add_argument("--no-auto-generate", action="store_true")
    production_build = production_sub.add_parser("build")
    production_build.add_argument("run_dir")
    production_qc = production_sub.add_parser("qc")
    production_qc.add_argument("production_run_manifest")
    production_metrics = production_sub.add_parser("metrics")
    production_metrics.add_argument("production_qc_manifest")
    production_report = production_sub.add_parser("report")
    production_report.add_argument("run_dir")

    features = sub.add_parser("features")
    features_sub = features.add_subparsers(dest="features_cmd")
    features_build = features_sub.add_parser("build")
    features_build.add_argument("paths", nargs="+")
    features_build.add_argument("--out", required=True)

    batch = sub.add_parser("batch")
    batch_sub = batch.add_subparsers(dest="batch_cmd")
    batch_plan = batch_sub.add_parser("plan")
    batch_plan.add_argument("formulations_csv")
    batch_plan.add_argument("--out", required=True)
    batch_smoke = batch_sub.add_parser("smoke")
    batch_smoke.add_argument("batch_plan")
    batch_smoke.add_argument("--workers", type=int, default=1)
    batch_smoke.add_argument("--dry-run", action="store_true")
    batch_sum = batch_sub.add_parser("summarize")
    batch_sum.add_argument("batch_dir")

    report = sub.add_parser("report")
    report_sub = report.add_subparsers(dest="report_cmd")
    report_run = report_sub.add_parser("run")
    report_run.add_argument("run_dir")
    report_batch = report_sub.add_parser("batch")
    report_batch.add_argument("batch_dir")

    audit = sub.add_parser("audit")
    audit_sub = audit.add_subparsers(dest="audit_cmd")
    audit_run = audit_sub.add_parser("run")
    audit_run.add_argument("run_dir")

    workflow = sub.add_parser("workflow", help="Run the complete MVP path from formulation to report")
    workflow.add_argument("input")
    workflow.add_argument("--out")
    workflow.add_argument("--real-gromacs", action="store_true", help="Use real GROMACS instead of dry-run smoke artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0
    try:
        if args.cmd == "env" and args.env_cmd == "doctor":
            core.command_env_doctor(args.json)
        elif args.cmd == "sources" and args.sources_cmd == "list":
            core.command_sources_list()
        elif args.cmd == "sources" and args.sources_cmd == "fetch":
            core.command_sources_fetch(args.all)
        elif args.cmd == "auto":
            core.command_auto(args.input, args.out, args.real_gromacs, args.allow_triage, args.policy)
        elif args.cmd == "intake":
            core.command_intake(args.input, args.out)
        elif args.cmd == "descriptors" and args.desc_cmd == "run":
            core.command_descriptors(args.intake_manifest)
        elif args.cmd == "topology" and args.topo_cmd == "index":
            core.command_topology_index(args.out)
        elif args.cmd == "topology" and args.topo_cmd == "resolve":
            core.command_topology_resolve(args.descriptor_manifest)
        elif args.cmd == "topology" and args.topo_cmd == "generate":
            core.command_topology_generate(args.descriptor_manifest)
        elif args.cmd == "topology" and args.topo_cmd == "validate":
            core.command_topology_validate(args.topology_files, args.out)
        elif args.cmd == "review" and args.review_cmd == "topology":
            core.command_review_topology(args.topology_candidates, args.answers)
        elif args.cmd == "templates" and args.templates_cmd == "list":
            core.command_templates_list()
        elif args.cmd == "templates" and args.templates_cmd == "recommend":
            core.command_templates_recommend(args.topology_review_manifest)
        elif args.cmd == "build" and args.build_cmd == "smoke":
            core.command_build_smoke(args.template_manifest, args.builder)
        elif args.cmd == "gromacs" and args.gmx_cmd == "preflight":
            core.command_gromacs_preflight(args.build_manifest)
        elif args.cmd == "simulate" and args.sim_cmd == "smoke":
            core.command_simulate_smoke(args.build_manifest, args.dry_run, args.gmx_extra)
        elif args.cmd == "qc" and args.qc_cmd == "smoke":
            core.command_qc_smoke(args.smoke_run_manifest)
        elif args.cmd == "metrics" and args.metrics_cmd == "extract":
            core.command_metrics_extract(args.qc_manifest, args.allow_failed_qc)
        elif args.cmd == "prioritize":
            core.command_prioritize(args.manifests, args.out)
        elif args.cmd == "production" and args.production_cmd == "plan":
            core.command_production_plan(args.run_dir, args.out, args.allow_placeholder)
        elif args.cmd == "production" and args.production_cmd == "run":
            core.command_production_run(args.run_dir, dry_run=not args.real_gromacs or args.dry_run, profile=args.profile, allow_placeholder=args.allow_placeholder, auto_generate=not args.no_auto_generate)
        elif args.cmd == "production" and args.production_cmd == "build":
            core.command_production_build(args.run_dir)
        elif args.cmd == "production" and args.production_cmd == "qc":
            core.command_production_qc(args.production_run_manifest)
        elif args.cmd == "production" and args.production_cmd == "metrics":
            core.command_production_metrics(args.production_qc_manifest)
        elif args.cmd == "production" and args.production_cmd == "report":
            core.command_production_report(args.run_dir)
        elif args.cmd == "features" and args.features_cmd == "build":
            core.command_features_build(args.paths, args.out)
        elif args.cmd == "batch" and args.batch_cmd == "plan":
            core.command_batch_plan(args.formulations_csv, args.out)
        elif args.cmd == "batch" and args.batch_cmd == "smoke":
            core.command_batch_smoke(args.batch_plan, args.dry_run)
        elif args.cmd == "batch" and args.batch_cmd == "summarize":
            core.command_batch_summarize(args.batch_dir)
        elif args.cmd == "report" and args.report_cmd == "run":
            core.command_report_run(args.run_dir)
        elif args.cmd == "report" and args.report_cmd == "batch":
            core.command_report_batch(args.batch_dir)
        elif args.cmd == "audit" and args.audit_cmd == "run":
            result = core.command_audit_run(args.run_dir)
            if result["status"] != "pass":
                return 1
        elif args.cmd == "workflow":
            core.command_workflow(args.input, args.out, dry_run=not args.real_gromacs)
        else:
            parser.error("unknown or incomplete command")
    except Exception as exc:
        print(f"automd failed: {exc}", file=sys.stderr)
        return 1
    return 0


app = main


if __name__ == "__main__":
    raise SystemExit(main())
