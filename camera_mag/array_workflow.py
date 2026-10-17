"""Auditable JSON -> native solve -> attribution -> Luna -> JSON loop."""
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from .array_config import ROOT, load_config, normalize_config, write_json
from .array_maxwell import digest, native_iteration, simulation_gate
from .array_search import accept_selection, agent_contract, candidate_search, improving_candidates, run_luna, utc
from .array_solver import CameraArraySolver
from .geometry import load_camera
from .materials import benchmark_materials


def make_solver(config, cache_dir):
    from .aedt_source import load_source_camera, sha256
    if "aedt_source" in config:
        if config.get("aedt_source_sha256") != sha256(config["aedt_source"]):
            raise ValueError("Frozen AEDT source changed before solver construction")
        camera, materials = load_source_camera(config["aedt_source"], config["low_fidelity"]["cell_mm"], cache_dir)
    elif "aedt" in config:
        raise ValueError("Extract and freeze the AEDT source before constructing a solver")
    else:
        camera = load_camera(config["step"], config["low_fidelity"]["cell_mm"], cache_dir)
        materials = benchmark_materials()
    return CameraArraySolver(camera, materials, cache_dir=cache_dir,
                             **{k: config["low_fidelity"][k] for k in ("tol", "max_iterations", "max_operator_mb")})


def analyze_layout(solver, config, folder, deadline):
    if perf_counter() >= deadline:
        raise TimeoutError("Campaign budget exhausted before low-fidelity solve")
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "config.json", config)
    solver.deadline = deadline
    report, solution = solver.evaluate(config)
    report["config_sha256"] = digest(folder / "config.json")
    write_json(folder / "analysis.json", report)
    np.savez_compressed(folder / "low_fidelity_state.npz", state=solution["state"], h=solution["h"], b=solution["b"],
                        positions_mm=[c["position_mm"] for c in config["cameras"]])
    if not solution["converged"]:
        raise RuntimeError("Low-fidelity baseline did not converge; raw state retained")
    return report, solution


def prepare_candidates(solver, config, report, solution, folder, deadline):
    search = candidate_search(solver, config, report, solution, deadline=deadline, record_path=folder / "candidates.json")
    search["config_sha256"] = report["config_sha256"]
    write_json(folder / "candidates.json", search)
    if improving_candidates(config, search):
        prompt, schema = agent_contract(config, report, search)
        agent = folder / "agent"
        agent.mkdir()
        (agent / "prompt.txt").write_text(prompt, encoding="utf-8")
        write_json(agent / "schema.json", schema)
    return search


def select_next(config, report, search, folder, deadline, *, proposal=None):
    if not improving_candidates(config, search):
        return None
    agent = folder / "agent"
    if proposal:
        selection = json.loads(Path(proposal).read_text(encoding="utf-8-sig"))
        write_json(agent / "supplied_answer.json", selection)
        provider = "supplied Luna response; caller is responsible for model provenance"
    else:
        remaining = deadline - perf_counter()
        if remaining < 10:
            raise TimeoutError("Insufficient campaign budget for Luna")
        prompt, schema = agent_contract(config, report, search)
        selection = run_luna(agent, prompt, schema, min(120, remaining))
        provider = "Codex CLI"
    updated, audit = accept_selection(config, search, selection)
    audit["provider"] = provider
    write_json(folder / "proposal_audit.json", audit)
    write_json(folder / "next.json", updated)
    return updated


def propose_prepared(output, *, proposal=None, budget_seconds=120):
    output = Path(output).resolve()
    if (output / "next.json").exists():
        raise FileExistsError(output / "next.json")
    config = load_config(output / "config.json")
    report = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    search = json.loads((output / "candidates.json").read_text(encoding="utf-8"))
    if report.get("config_sha256") != digest(output / "config.json") or search.get("config_sha256") != report["config_sha256"]:
        raise ValueError("Prepared configuration changed or lacks provenance; use a fresh prepare directory")
    if not np.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("Invalid proposal budget")
    start = perf_counter()
    try:
        result = select_next(config, report, search, output, start + budget_seconds, proposal=proposal)
        return {"status": "proposed" if result else "no_improving_candidate", "next": str(output / "next.json") if result else None}
    finally:
        write_json(output / "propose_timing.json", {"seconds": perf_counter() - start})


def run_workflow(config_path, output, *, mode="prepare", surrogate_only=False, max_iterations=2,
                 budget_minutes=5, estimated_minutes=None, estimate_basis=None, confirmed_long=False,
                 cache_dir=".cache", proposal=None):
    config, output = load_config(config_path), Path(output).resolve()
    if mode not in ("prepare", "build", "run"):
        raise ValueError("Unsupported workflow mode")
    if surrogate_only and mode != "run":
        raise ValueError("surrogate_only is valid only for run; use prepare for offline analysis")
    if output.exists():
        raise FileExistsError("Use a fresh output directory; previous projects and failures are never overwritten")
    gate = None
    if mode == "run" and not surrogate_only:
        gate = simulation_gate(estimated_minutes=estimated_minutes, estimate_basis=estimate_basis,
                               budget_minutes=budget_minutes, confirmed_long=confirmed_long)
    if max_iterations < 0 or not np.isfinite(budget_minutes) or budget_minutes <= 0:
        raise ValueError("Invalid iteration count or campaign budget")
    if mode == "build" and budget_minutes > 30:
        raise ValueError("Build-only mode has a maximum 30-minute budget")
    output.mkdir(parents=True)
    start, started = perf_counter(), utc()
    deadline = start + budget_minutes * 60
    record = {"started_UTC": started, "mode": mode, "surrogate_only": surrogate_only,
              "simulation_gate": gate, "budget_minutes": budget_minutes, "max_moves": max_iterations,
              "iterations": [], "native_jobs_submitted": 0, "input": str(Path(config_path).resolve())}
    best, best_score = None, float("inf")
    try:
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        sources = list((ROOT / "camera_mag").glob("*.py")) + [ROOT / "scripts/camera_interference.py", ROOT / "scripts/run_camera_array_native.py", ROOT / "scripts/extract_camera_aedt.py"]
        hashes = {}
        for source in sources:
            target = snapshot / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(source.relative_to(ROOT))] = digest(target)
        write_json(snapshot / "sha256.json", hashes)
        if "aedt" in config:
            from .aedt_source import extract_source
            source = extract_source(config.pop("aedt"), output / "aedt_source", deadline)
            config["aedt_source"] = str(source)
            config = normalize_config(config)
        solver = make_solver(config, cache_dir)
        for iteration in range(max_iterations + 1 if mode == "run" else 1):
            folder = output / f"iteration_{iteration:03d}" if mode == "run" else output
            item = {"iteration": iteration, "directory": str(folder), "started_UTC": utc()}
            record["iterations"].append(item)
            write_json(output / "run_record.json", record)
            report, solution = analyze_layout(solver, config, folder, deadline)
            item["low_fidelity_score"] = report["score"]
            if mode == "build" or (mode == "run" and not surrogate_only):
                native = native_iteration(folder / "config.json", folder, deadline, build_only=mode == "build")
                if mode == "build":
                    record["status"] = "built_unsolved"
                    break
                record["native_jobs_submitted"] += 1
                item["native_score"] = native["score"]
                item["native_adaptive_converged"] = native["native_adaptive_converged"]
                if not native["native_adaptive_converged"]:
                    record["status"] = "native_unconverged_stop"
                    break
                score = native["score"]
            else:
                score = report["score"]
            if mode == "run":
                if score >= best_score:
                    record["status"] = "no_verified_improvement_stop" if not surrogate_only else "no_surrogate_improvement_stop"
                    break
                best, best_score = config, score
                write_json(output / "best.json", best)
                record["best_score"] = score
                record["best_iteration"] = iteration
                record["best_fidelity"] = "low-fidelity" if surrogate_only else "Maxwell"
                if score <= config["search"]["stop_score"]:
                    record["status"] = "objective_reached"
                    break
                if iteration == max_iterations:
                    record["status"] = "iteration_limit"
                    break
            search = prepare_candidates(solver, config, report, solution, folder, deadline)
            item["candidate_count"] = len(search["candidates"])
            if mode == "prepare":
                record["status"] = "prepared" if improving_candidates(config, search) else "no_improving_candidate"
                break
            updated = select_next(config, report, search, folder, deadline, proposal=proposal)
            if updated is None:
                record["status"] = "no_improving_candidate"
                break
            config = updated
            # A supplied response covers one iteration only; subsequent steps
            # use the actual Luna CLI, never silently reuse an old decision.
            proposal = None
            write_json(output / "run_record.json", record)
        return record
    except BaseException as exc:
        record.update(status="stopped", failure=repr(exc))
        raise
    finally:
        record["native_jobs_submitted"] = len(list(output.rglob("native_solve.log.json")))
        record.update(ended_UTC=utc(), elapsed_seconds=perf_counter() - start,
                      best_native_verified=best is not None and mode == "run" and not surrogate_only)
        write_json(output / "run_record.json", record)
