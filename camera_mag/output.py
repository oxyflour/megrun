import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .coil import coil_forces
from .features import feature_table


def write_vtk(table, path):
    """Legacy point-cloud VTK, coordinates in metres; these are not FEM cells."""
    with Path(path).open("w", encoding="ascii") as f:
        n = len(table)
        f.write(f"# vtk DataFile Version 3.0\nCoarse magnetic samples (SI units)\nASCII\nDATASET POLYDATA\nPOINTS {n} double\n")
        np.savetxt(f, table[["x", "y", "z"]].to_numpy(), fmt="%.10g")
        f.write(f"VERTICES {n} {2*n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")
        f.write(f"POINT_DATA {n}\n")
        for name in ["Bmag", "Hmag", "saturation", "gradB", "refinement_level", "residual"]:
            f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, table[name].to_numpy(), fmt="%.10g")
        for prefix in ["B", "H", "M"]:
            f.write(f"VECTORS {prefix} double\n")
            np.savetxt(f, table[[prefix + axis for axis in "xyz"]].to_numpy(), fmt="%.10g")


def field_plot(table, path, distance_mm):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for col, (value, title) in enumerate([("Bmag", "|B| [T]"), ("Hmag", "|H| [A/m]"),
                                         ("saturation", "Yoke B / 1.6 T"), ("refinement_level", "Heuristic refinement level")]):
        for row, plane in enumerate([("x", "y"), ("x", "z")]):
            ax = axes[row, col]
            # Use a center slice for XZ; XY projection is intentionally all cells.
            view = table if row == 0 else table[np.abs(table.y) < .00075]
            scatter = ax.scatter(view[plane[0]] * 1e3, view[plane[1]] * 1e3, c=view[value], s=9,
                                 cmap="viridis", vmin=table[value].min(), vmax=table[value].max())
            fig.colorbar(scatter, ax=ax, shrink=.8)
            ax.set(xlabel=plane[0] + " [mm]", ylabel=plane[1] + " [mm]", title=title)
            ax.set_aspect("equal")
    fig.suptitle(f"Low-fidelity camera pair: outer-envelope gap {distance_mm:g} mm")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_result(solver, result, directory, isolated=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    isolated = solver.isolated_states() if isolated is None else isolated
    forces = coil_forces(solver, result, isolated)
    table = feature_table(solver, result)
    table.to_parquet(directory / "features.parquet", index=False)
    table.to_csv(directory / "features.csv", index=False)
    np.savez_compressed(directory / "state.npz", distance_mm=result.distance_mm, offset_b=result.offset_b,
                        M_A=result.state[0], M_B=result.state[1], H_A=result.h[0], H_B=result.h[1],
                        B_A=result.b[0], B_B=result.b[1], centers_A=solver.camera_a.centers,
                        centers_B=solver.camera_b.centers + result.offset_b,
                        residual_history=np.array(result.residual_history))
    write_vtk(table, directory / "field.vtk")
    field_plot(table, directory / "field.png", result.distance_mm)
    recommendations = {}
    for (module, name), group in table.groupby(["module", "object"]):
        level = int(np.ceil(np.quantile(group.refinement_level, .9)))
        recommendations[module + "/" + name] = {"refinement_level": level,
              "max_element_length_mm": solver.camera_a.metadata["cell_mm"] / 2 ** (level + 1)}
    (directory / "mesh_recommendations.json").write_text(json.dumps({"source": "untrained physics heuristic",
            "units": "mm", "operations": recommendations}, indent=2), encoding="utf-8")
    normalized_residual = np.linalg.norm(np.concatenate(result.constitutive_residual)) / max(np.linalg.norm(np.concatenate(result.state)), 1)
    summary = {"fidelity": "coarse voxel-dipole diagnostic; not Maxwell reference force",
               "distance_definition": "outer-envelope gap along +X; identical orientation",
               "distance_mm": result.distance_mm, "center_offset_mm": (result.offset_b * 1e3).tolist(),
               "current_A": solver.current_a, "turns": solver.turns,
               "winding_direction": "counterclockwise viewed from +Z for positive current",
               "coil_force": forces, "converged": result.converged, "nonlinear_iterations": result.iterations,
               "preconditioned_residual": result.residual_history[-1], "constitutive_relative_residual": float(normalized_residual),
               "residual_history": result.residual_history, "runtime": result.runtime,
               "self_operator_cache": solver.self_cache, "self_precompute_calls": solver.self_build_calls,
               "isolated_solve": solver.isolated_runtime,
               "relative_M_change": {label: float(np.linalg.norm(mi - iso) / max(np.linalg.norm(iso), 1))
                   for label, mi, iso in zip("AB", result.state, isolated)},
               "geometry": solver.camera_a.metadata,
               "assumptions": ["Illustrative steel B-H curve and NdFeB Br=1.2 T unless configured",
                               "Gaussian volume-averaged dipoles; N=1/3 self term; point-dipole far field",
                               "Uniform turns across rectangular copper cross-section; regularized current source",
                               "Coil force omits its own current field; includes both coils' magnetic backreaction",
                               "No mesh or force accuracy certification; refinement levels are an untrained heuristic"]}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
