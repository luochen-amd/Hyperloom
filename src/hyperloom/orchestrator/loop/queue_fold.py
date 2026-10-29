# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fold the untested-proposal queue head into the explore grid a delegate is seeding.

The idle-lane bench (``FrameworkPhase._maybe_bench_untested_proposals``) needs an
explore-free lane, and the Coordinator's own grids hold that lane back-to-back,
so a specialist's best row can spend most of the session waiting for its benchmark
slot. Folding the head rows into the grid the Coordinator is seeding anyway moves
their measurement into that grid -- same lane, no reordering of the queue, no
pre-emption -- and the rows leave the queue exactly as a benched row does, through
the explore round recording them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["fold_queue_head_into_grid"]

#: Rows folded per grid: the queue's oldest debt rides along without crowding the
#: Coordinator's own choices out of the grid cap.
FOLD_CAP = 2


def _grid_fingerprints(tasks: Any) -> set[str]:
    """Content fingerprints of every variant in a queued or running explore grid."""
    from ..actions.executors._proposal_identity import content_fingerprint

    return {
        content_fingerprint(variant)
        for task in (*tasks.queued_context_sync(), *tasks.running_context_sync())
        if task.kind == "explore"
        for variant in (task.params or {}).get("grid") or ()
        if isinstance(variant, dict)
    }


def _variant_of(row: dict[str, Any]) -> dict[str, Any]:
    """The variant shape an auto-benched proposal row carries (``_enqueue_bench_grid``)."""
    from ..actions.executors._proposal_identity import controls_of

    return {
        "name": row["name"],
        "extra_args": row["extra_args"],
        "extra_envs": dict(row["extra_envs"]),
        **controls_of(row),
        "provenance": row["provenance"],
        # The proposing specialist's whole reasoning and checked citations reach the measured Experience.
        "reasoning": row["reason"],
        **({"experience_citations": row["experience_citations"]} if row["experience_citations"] else {}),
    }


def fold_queue_head_into_grid(params: dict[str, Any], state: Any, tasks: Any) -> int:
    """Prepend up to ``FOLD_CAP`` untested queue rows to ``params['grid']``; returns how many folded.

    The queue is already benched-free (``untested_proposal_rows`` filters benched
    fingerprints), so a row is skipped only when some grid already holds it --
    a queued or running one, or the grid being seeded, whose head rows the
    Coordinator may have copied itself. Custom (operator) rows are left to the
    operator-priority bench that owns them.
    """
    from ..actions.executors._proposal_identity import content_fingerprint
    from ..phases.proposal_bench import CUSTOM_PROPOSAL_DOMAIN

    grid = params.get("grid")
    if not isinstance(grid, list) or not grid:
        return 0
    held = _grid_fingerprints(tasks) | {content_fingerprint(variant) for variant in grid if isinstance(variant, dict)}
    to_fold = [
        row
        for row in state.untested_proposal_rows()
        if row["domain"] != CUSTOM_PROPOSAL_DOMAIN and row["fingerprint"] not in held
    ][:FOLD_CAP]
    params["grid"] = [*map(_variant_of, to_fold), *grid]
    return len(to_fold)
