"""Report preprocessing losses without changing structures or metric inputs."""

from collections import Counter

import numpy as np

from lemat_genbench.preprocess.base import PreprocessorResult


def preprocessor_coverage(result: PreprocessorResult) -> dict:
    """Summarize a preprocessor's output before any validity filtering."""
    return {
        "n_input": result.n_input_structures,
        "n_output": len(result.processed_structures),
        "failed_indices": list(result.failed_indices),
        "warnings": list(result.warnings),
    }


def _error_reason(value) -> str:
    lines = str(value).splitlines()
    return lines[0][:120] if lines else ""


def _non_finite(value) -> bool:
    # Count a vector once if any of its components is NaN or infinite.
    try:
        return not np.isfinite(value).all()
    except (TypeError, ValueError):
        # Non-numeric properties are present, but are not NaN/inf values.
        return False


def mlip_coverage(structures, mlip_names, relaxed: bool, embeddings: bool) -> dict:
    """Count present values, failures and non-finite values from properties only.

    Failure flags count true values. Numeric and embedding quantities count every
    non-None value, including non-finite values, which are also reported separately.
    """
    quantities = ["energy"]
    if relaxed:
        quantities.extend(
            [
                "formation_energy",
                "e_above_hull",
                "relaxation_rmse",
                "relaxed_e_above_hull",
            ]
        )
    if embeddings:
        quantities.append("graph_embedding")

    coverage = {"n_structures": len(structures), "models": {}}
    for model in mlip_names:
        counts = dict.fromkeys(quantities, 0)
        non_finite = Counter()
        model_errors = Counter()
        relaxation_errors = Counter()
        relaxation_failed = 0
        for structure in structures:
            properties = structure.properties
            for quantity in quantities:
                value = properties.get(f"{quantity}_{model}")
                if value is not None:
                    counts[quantity] += 1
                    if _non_finite(value):
                        non_finite[quantity] += 1
            error = properties.get(f"mlip_error_{model}")
            if error is not None:
                model_errors[_error_reason(error)] += 1
            if relaxed:
                relaxation_failed += bool(properties.get(f"relaxation_failed_{model}"))
                error = properties.get(f"relaxation_error_{model}")
                if error is not None:
                    relaxation_errors[_error_reason(error)] += 1

        if relaxed:
            counts["relaxation_failed"] = relaxation_failed
            counts["relaxation_errors"] = dict(relaxation_errors)
        counts["model_errors"] = dict(model_errors)
        counts["non_finite"] = dict(non_finite)
        coverage["models"][model] = counts

    if relaxed:
        histogram = dict.fromkeys(map(str, range(len(mlip_names) + 1)), 0)
        for structure in structures:
            n_mlips = structure.properties.get("e_above_hull_n_mlips")
            histogram[str(n_mlips if n_mlips is not None else 0)] += 1
        coverage["e_above_hull_n_mlips"] = histogram
    return coverage


def coverage_warnings(coverage: dict) -> list[str]:
    """Return one summary per step/model/quantity gap, with no score changes."""
    warnings = []
    steps = dict(coverage.get("preprocessors", {}))
    if "validity" in coverage:
        steps = {"validity": coverage["validity"], **steps}
    for step, counts in steps.items():
        n_input = counts["n_input"]
        n_dropped = max(n_input - counts["n_output"], len(counts["failed_indices"]))
        if n_dropped:
            description = (
                "raised during the validity check" if step == "validity" else "dropped"
            )
            warnings.append(
                f"[{step}] {n_dropped} of {n_input} structures {description}"
            )
        if counts["warnings"]:
            warnings.append(f"[{step}] {len(counts['warnings'])} preprocessor warnings")

    mlip = coverage.get("multi_mlip", {})
    n_structures = mlip.get("n_structures", 0)
    models = mlip.get("models", {})
    for model, counts in models.items():
        prefix = f"[multi_mlip] {model}:"
        for quantity, count in counts.items():
            if quantity in {
                "relaxation_failed",
                "relaxation_errors",
                "model_errors",
                "non_finite",
            }:
                continue
            if count < n_structures:
                warnings.append(
                    f"{prefix} {quantity} missing for {n_structures - count} "
                    f"of {n_structures} structures"
                )
        n_failed = counts.get("relaxation_failed", 0)
        reasons = counts.get("relaxation_errors", {})
        if n_failed or reasons:
            details = ", ".join(f"{reason}: {n}" for reason, n in reasons.items())
            warnings.append(
                f"{prefix} relaxation failed for {n_failed} of {n_structures} structures"
                + (f" ({details})" if details else "")
            )
        reasons = counts.get("model_errors", {})
        if reasons:
            details = ", ".join(f"{reason}: {n}" for reason, n in reasons.items())
            warnings.append(
                f"{prefix} model errors for {sum(reasons.values())} "
                f"of {n_structures} structures ({details})"
            )
        for quantity, count in counts.get("non_finite", {}).items():
            if count:
                warnings.append(
                    f"{prefix} {quantity} non-finite for {count} "
                    f"of {n_structures} structures"
                )

    histogram = mlip.get("e_above_hull_n_mlips", {})
    incomplete = {
        n: count for n, count in histogram.items() if int(n) < len(models) and count
    }
    if incomplete:
        details = ", ".join(f"{n}: {count}" for n, count in incomplete.items())
        warnings.append(
            f"[multi_mlip] e_above_hull from fewer than {len(models)} models for "
            f"{sum(incomplete.values())} of {n_structures} structures ({details})"
        )
    return warnings
