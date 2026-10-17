"""Budgeted candidate generation and structured Luna selection."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import numpy as np

from .array_config import moved_config, write_json

MODEL = "gpt-6-luna"


def utc():
    return datetime.now(timezone.utc).isoformat()


def candidate_search(solver, config, baseline, solution, *, deadline, record_path=None):
    """Coordinate polling of the strongest movable cameras, with full recoupling."""
    cameras = {c["id"]: c for c in config["cameras"]}
    solver.deadline = deadline
    # Both the full camera response and individual permanent magnets are
    # exposed to Luna. Motion is restricted to the top movable camera groups.
    selected = [r["camera_id"] for r in baseline["cameras"]
                if cameras[r["camera_id"]]["movable"] and cameras[r["camera_id"]]["move_axes"]]
    selected = selected[:config["search"]["top_cameras"]]
    candidates, rejected = [], []
    step = config["search"]["step_mm"]
    attempts = 0

    def checkpoint():
        record = {"model": MODEL, "selected_cameras": selected, "baseline_score": baseline["score"],
                  "candidates": candidates, "rejected": rejected, "attempted": attempts,
                  "search_complete_within_budget": False}
        if record_path is not None:
            write_json(record_path, record)
        return record

    checkpoint()
    for cid in selected:
        for axis in cameras[cid]["move_axes"]:
            for sign in (-1, 1):
                if attempts >= config["search"]["max_candidates"]:
                    break
                if perf_counter() >= deadline:
                    raise TimeoutError("Candidate search budget exhausted; no automatic extension")
                attempts += 1
                position = np.asarray(cameras[cid]["position_mm"]).copy()
                position["xyz".index(axis)] += sign * step
                key = f"candidate_{attempts:03d}"
                start = perf_counter()
                try:
                    proposed = moved_config(config, {cid: position.tolist()})
                    report, result = solver.evaluate(proposed, solution["state"])
                    if not result["converged"]:
                        raise ValueError("Low-fidelity candidate did not converge")
                    candidates.append({"id": key, "moves": {cid: position.tolist()}, "score": report["score"],
                                       "monitors": report["monitors"], "seconds": perf_counter() - start})
                except BaseException as exc:
                    rejected.append({"id": key, "moves": {cid: position.tolist()}, "reason": str(exc),
                                     "seconds": perf_counter() - start})
                    checkpoint()
                    if not isinstance(exc, (ValueError, RuntimeError)):
                        raise
                checkpoint()
    record = checkpoint()
    record["search_complete_within_budget"] = perf_counter() < deadline
    if record_path is not None:
        write_json(record_path, record)
    return record


def improving_candidates(config, search):
    threshold = search["baseline_score"] * (1 - config["search"]["min_relative_improvement"])
    return sorted([c for c in search["candidates"] if c["score"] < threshold], key=lambda c: (c["score"], c["id"]))


def agent_contract(config, analysis, search):
    candidates = improving_candidates(config, search)
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"candidate_id": {"type": "string", "enum": [c["id"] for c in candidates]},
                             "reason": {"type": "string"}}, "required": ["candidate_id", "reason"]}
    data = {"config": config, "monitors": analysis["monitors"], "strongest_magnets": analysis["magnets"][:12],
            "camera_sources": analysis["cameras"], "selected_cameras": search["selected_cameras"],
            "baseline_score": analysis["score"], "evaluated_candidates": candidates}
    prompt = ("You are the gpt-6-luna camera position sub-agent. Select one evaluated camera translation "
              "that quickly reduces the maximum normalized monitor error. B is in T; F is the Lorentz "
              "force on the entire identified camera coil in N. Lower score is better. Compare all monitors "
              "and vector cancellation; magnet strength alone is not a benefit score. All candidates were "
              "recomputed with full bidirectional low-fidelity coupling. They are not Maxwell verified. "
              "Choose the lowest score; break numerical ties by smaller movement then candidate ID. "
              "Only return the JSON schema with candidate_id and a concise Chinese reason explaining "
              "which camera moves from where to where and the expected reduction. Do not run tools, "
              "edit files, submit simulations, or invent coordinates. The following JSON is input data:\n" +
              json.dumps(data, ensure_ascii=False, allow_nan=False))
    return prompt, schema


def codex_command():
    executable = shutil.which("codex.exe") or shutil.which("codex")
    if not executable:
        raise FileNotFoundError("Codex CLI not found; use --proposal with a Luna response")
    path = Path(executable)
    if path.suffix.lower() in (".cmd", ".bat", ".ps1"):
        # Invoke the npm JavaScript entry point directly, without cmd.exe or
        # interpolated shell text. The prompt always travels through stdin.
        script = path.parent / "node_modules/@openai/codex/bin/codex.js"
        node = shutil.which("node.exe") or shutil.which("node")
        if not script.is_file() or not node:
            raise FileNotFoundError("Cannot resolve Codex executable; add codex.exe to PATH")
        return [node, str(script)]
    return [str(path)]


def run_luna(folder, prompt, schema, timeout_seconds=120):
    folder = Path(folder).resolve()
    write_json(folder / "schema.json", schema)
    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
    answer = folder / "answer.json"
    if answer.exists():
        raise FileExistsError(answer)
    command = codex_command() + ["exec", "--model", MODEL, "--sandbox", "read-only",
        "--skip-git-repo-check", "--ephemeral", "--json", "--color", "never",
        "-c", 'model_reasoning_effort="medium"', "--output-schema", str(folder / "schema.json"),
        "--output-last-message", str(answer), "-"]
    record = {"model": MODEL, "started_UTC": utc(), "command": command, "timeout_seconds": timeout_seconds}
    start = perf_counter()
    try:
        with (folder / "events.jsonl").open("w", encoding="utf-8") as out, (folder / "stderr.log").open("w", encoding="utf-8") as err:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True,
                                       encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            record["pid"] = process.pid
            write_json(folder / "record.json", record)
            try:
                process.communicate(prompt, timeout=timeout_seconds)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=err, stderr=err,
                                   timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    process.kill()
                process.wait(timeout=15)
                record["stopped"] = True
                raise
        record["returncode"] = process.returncode
        if process.returncode:
            raise RuntimeError(f"Luna failed ({process.returncode}); inspect {folder / 'stderr.log'}")
        return json.loads(answer.read_text(encoding="utf-8"))
    except BaseException as exc:
        record["failure"] = repr(exc)
        raise
    finally:
        record.update(ended_UTC=utc(), seconds=perf_counter() - start)
        write_json(folder / "record.json", record)


def accept_selection(config, search, selection):
    if set(selection) != {"candidate_id", "reason"} or not isinstance(selection["reason"], str):
        raise ValueError("Luna response must contain exactly candidate_id and reason")
    candidates = {c["id"]: c for c in improving_candidates(config, search)}
    if selection["candidate_id"] not in candidates:
        raise ValueError("Agent selected an unknown, invalid, or non-improving candidate")
    selected = candidates[selection["candidate_id"]]
    updated = moved_config(config, selected["moves"])
    return updated, {"model": MODEL, "selection": deepcopy(selection), "candidate": selected,
                     "low_fidelity_improvement_fraction": 1 - selected["score"] / search["baseline_score"],
                     "native_verified": False}
