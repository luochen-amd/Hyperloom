# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The fold that rides untested proposal rows along on the Coordinator's explore grid.

``_seed_grid_params`` prepends the untested queue's head rows to the grid a
delegate carries, so a specialist's best proposal is measured by the next grid
the lane runs instead of waiting for an explore-free window the Coordinator's
own grids rarely leave. The rows leave the queue exactly as a benched row does,
so the fold must not duplicate a variant some grid already holds.
"""

from __future__ import annotations

from types import SimpleNamespace

from hyperloom.orchestrator.actions.executors._proposal_identity import content_fingerprint
from hyperloom.orchestrator.loop.queue_fold import FOLD_CAP, fold_queue_head_into_grid
from hyperloom.orchestrator.phases.proposal_bench import CUSTOM_PROPOSAL_DOMAIN


def _row(name: str, arg: str, *, domain: str = "freeform", order: int = 1) -> dict[str, object]:
    """A queue row the rank order of ``untested_proposal_rows`` would sort by ``order``."""
    extra_args = [f"--x={arg}"]
    return {
        "name": name,
        "extra_args": extra_args,
        "extra_envs": {"E": "1"},
        "remove_args": (),
        "unset_envs": (),
        "args_mode": "append",
        "atomic": True,
        "reason": "why",
        "domain": domain,
        "severity": 1,
        "provenance": "specialist",
        "experience_citations": ["k1"],
        "kb_read_id": "",
        "kb_rendered_refs": [],
        "fingerprint": content_fingerprint({"name": name, "extra_args": extra_args, "extra_envs": {"E": "1"}}),
        "order": order,
        "priority": 1,
    }


def _tasks(*grids: tuple[str, list[dict[str, object]]]) -> SimpleNamespace:
    """A task registry whose queued/running lanes hold the given ``(lane, grid)`` explore tasks."""
    held = [SimpleNamespace(kind="explore", params={"grid": grid}) for _, grid in grids]
    queued = [t for (lane, _), t in zip(grids, held) if lane == "queued"]
    running = [t for (lane, _), t in zip(grids, held) if lane == "running"]
    return SimpleNamespace(queued_context_sync=lambda: queued, running_context_sync=lambda: running)


def test_the_head_rows_ride_along_and_custom_rows_do_not():
    llm_choice = {"name": "llm-choice", "extra_args": ["--z"], "extra_envs": {}}
    params = {"grid": [llm_choice]}
    state = SimpleNamespace(
        untested_proposal_rows=lambda: [
            _row("custom-row", "9", domain=CUSTOM_PROPOSAL_DOMAIN, order=2),
            _row("free-row", "1", order=1),
            _row("third-row", "3", order=3),
        ]
    )

    folded = fold_queue_head_into_grid(params, state, _tasks())

    assert folded == 2
    assert [variant["name"] for variant in params["grid"]] == ["free-row", "third-row", "llm-choice"]
    assert params["grid"][0]["provenance"] == "specialist"
    assert params["grid"][0]["reasoning"] == "why"
    assert params["grid"][0]["experience_citations"] == ["k1"]


def test_a_row_a_grid_already_holds_is_not_folded_twice():
    held = [{"name": "free-row", "extra_args": ["--x=1"], "extra_envs": {"E": "1"}}]
    llm_choice = {"name": "llm-choice", "extra_args": ["--z"], "extra_envs": {}}
    state = SimpleNamespace(untested_proposal_rows=lambda: [_row("free-row", "1"), _row("third-row", "3", order=3)])

    for lane in ("queued", "running"):
        params = {"grid": [llm_choice]}
        assert fold_queue_head_into_grid(params, state, _tasks((lane, held))) == 1
        assert params["grid"][0]["name"] == "third-row"

    # The grid being seeded holds it too: folding again adds nothing.
    params = {"grid": [held[0]]}
    assert fold_queue_head_into_grid(params, state, _tasks()) == 1
    assert fold_queue_head_into_grid(params, state, _tasks()) == 0


def test_only_the_cap_rides_along():
    rows = [_row(f"r{i}", str(100 + i), order=i) for i in range(2 * FOLD_CAP + 1)]
    params = {"grid": [{"name": "llm-choice", "extra_args": ["--z"], "extra_envs": {}}]}

    assert fold_queue_head_into_grid(params, SimpleNamespace(untested_proposal_rows=lambda: rows), _tasks()) == FOLD_CAP
    assert len(params["grid"]) == FOLD_CAP + 1


def test_without_a_grid_nothing_is_folded():
    state = SimpleNamespace(untested_proposal_rows=lambda: [_row("free-row", "1")])

    for params in ({}, {"grid": []}, {"grid": None}, {"grid": "no"}):
        before = dict(params)
        assert fold_queue_head_into_grid(params, state, _tasks()) == 0
        assert params == before
