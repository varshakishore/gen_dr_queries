"""Mock-based tests for round-0 eval and multi-criteria judging in process_seed."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import research_pipeline as rp


def _fake_claude_response(payload: dict) -> tuple[str, rp.CostBucket]:
    return json.dumps(payload), rp.CostBucket(calls=1)


def run_scenario(name: str, claude_responses_by_purpose, answers_by_attempt):
    call_counts = {"seed_criterion": 0, "harder": 0, "judge": 0}

    def fake_call_claude(client, *, model, system, messages, max_tokens,
                        logger, seed, attempt, purpose):
        idx = call_counts[purpose]
        call_counts[purpose] += 1
        payload = claude_responses_by_purpose[purpose][idx]
        return _fake_claude_response(payload)

    def fake_query(question, logger, seed, attempt, url=None, timeout_s=None):
        return answers_by_attempt[attempt]

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "log.jsonl"
        logger = rp.RunLogger(log_path, run_id="test")
        with patch.object(rp, "_call_claude", side_effect=fake_call_claude), \
             patch.object(rp, "query_research_system", side_effect=fake_query):
            result = rp.process_seed(
                client=None, model="claude-sonnet-4-5",
                seed="What is photosynthesis?",
                logger=logger, max_attempts=3, verbose=False,
            )
    print(f"\n=== Scenario: {name} ===")
    print(f"final_status: {result.final_status}")
    print(f"num attempts recorded: {len(result.attempts)}")
    for a in result.attempts:
        n_crit = len(a.harder.verification_criteria)
        print(f"  attempt={a.attempt} verdict={a.judgment.verdict} "
              f"failed={a.judgment.criteria_failed_count}/{n_crit} "
              f"question={a.harder.updated_question!r}")
    return result


def _judge_payload(results: list[dict], verdict: str | None = None) -> dict:
    """Build a judge response: results is a list of {criterion, satisfied[, reasoning]}.
    If verdict is None, derive it from the satisfied flags."""
    failed = sum(1 for r in results if not r.get("satisfied", True))
    if verdict is None:
        verdict = "FAILED" if failed > 0 else "PASSED"
    return {
        "criterion_results": [
            {"criterion": r["criterion"],
             "satisfied": r["satisfied"],
             "reasoning": r.get("reasoning", "ok")}
            for r in results
        ],
        "criteria_failed_count": failed,
        "other_issues": [],
        "summary": "test summary",
        "verdict": verdict,
    }


# ---------- Scenario 1: seed is already hard (any criterion failure → stop) ----------
seed_crit = ["Must explain the Calvin cycle.",
             "Must cite at least 2 academic sources.",
             "Must compare C3 vs C4 plants."]
r1 = run_scenario(
    "seed already hard (1/3 criteria fail)",
    claude_responses_by_purpose={
        "seed_criterion": [{"verification_criteria": seed_crit}],
        "harder": [],
        "judge": [_judge_payload([
            {"criterion": seed_crit[0], "satisfied": True},
            {"criterion": seed_crit[1], "satisfied": False, "reasoning": "no citations"},
            {"criterion": seed_crit[2], "satisfied": True},
        ])],
    },
    answers_by_attempt={0: "Photosynthesis is when plants eat sunlight."},
)
assert r1.final_status == "ALREADY_HARD", r1.final_status
assert len(r1.attempts) == 1
assert r1.attempts[0].attempt == 0
assert r1.attempts[0].harder.verification_criteria == seed_crit
assert r1.attempts[0].judgment.criteria_failed_count == 1
assert r1.attempts[0].judgment.verdict == "FAILED"
print("Scenario 1 PASSED")


# ---------- Scenario 2: seed passes (0/2 fail), round 1 harder fails (3/3) ----------
seed_crit2 = ["Must explain Calvin cycle.", "Must mention chlorophyll."]
harder_crit = ["Must synthesize biology + economics.",
               "Must cite 5+ sources.",
               "Must consider conflicting viewpoints."]
r2 = run_scenario(
    "seed easy, round 1 fails 3/3",
    claude_responses_by_purpose={
        "seed_criterion": [{"verification_criteria": seed_crit2}],
        "harder": [{
            "brainstorming": "...",
            "chosen_strategy": "cross-domain synthesis",
            "updated_question": "How does photosynthesis tie to global carbon markets?",
            "why_harder": "spans biology + econ",
            "verification_criteria": harder_crit,
        }],
        "judge": [
            _judge_payload([
                {"criterion": seed_crit2[0], "satisfied": True},
                {"criterion": seed_crit2[1], "satisfied": True},
            ]),
            _judge_payload([
                {"criterion": harder_crit[0], "satisfied": False},
                {"criterion": harder_crit[1], "satisfied": False},
                {"criterion": harder_crit[2], "satisfied": False},
            ]),
        ],
    },
    answers_by_attempt={
        0: "Photosynthesis converts light into chemical energy via the Calvin cycle, with chlorophyll absorbing the light.",
        1: "Plants make sugar.",
    },
)
assert r2.final_status == "FAILED_FOUND", r2.final_status
assert len(r2.attempts) == 2
assert r2.attempts[0].judgment.verdict == "PASSED"
assert r2.attempts[0].judgment.criteria_failed_count == 0
assert r2.attempts[1].judgment.verdict == "FAILED"
assert r2.attempts[1].judgment.criteria_failed_count == 3
assert r2.attempts[1].harder.verification_criteria == harder_crit
print("Scenario 2 PASSED")


# ---------- Scenario 3: everything passes, exhausts attempts ----------
r3 = run_scenario(
    "everything passes",
    claude_responses_by_purpose={
        "seed_criterion": [{"verification_criteria": ["Must explain Calvin cycle."]}],
        "harder": [
            {"brainstorming": "...", "chosen_strategy": f"strategy {i}",
             "updated_question": f"harder q {i}",
             "why_harder": "...", "verification_criteria": [f"crit {i}-a", f"crit {i}-b"]}
            for i in range(1, 4)
        ],
        "judge": [
            _judge_payload([{"criterion": "Must explain Calvin cycle.", "satisfied": True}]),
            _judge_payload([{"criterion": "crit 1-a", "satisfied": True},
                            {"criterion": "crit 1-b", "satisfied": True}]),
            _judge_payload([{"criterion": "crit 2-a", "satisfied": True},
                            {"criterion": "crit 2-b", "satisfied": True}]),
            _judge_payload([{"criterion": "crit 3-a", "satisfied": True},
                            {"criterion": "crit 3-b", "satisfied": True}]),
        ],
    },
    answers_by_attempt={i: f"answer {i}" for i in range(4)},
)
assert r3.final_status == "EXHAUSTED", r3.final_status
assert len(r3.attempts) == 4
assert [a.attempt for a in r3.attempts] == [0, 1, 2, 3]
assert all(a.judgment.criteria_failed_count == 0 for a in r3.attempts)
print("Scenario 3 PASSED")


# ---------- Scenario 4: seed barely passes (0 fail), but later round fails just 1/3 ----------
# Confirms "any failure is sufficient signal" — 1 failure is enough to FAIL.
r4 = run_scenario(
    "round 2 fails only 1/3 criteria",
    claude_responses_by_purpose={
        "seed_criterion": [{"verification_criteria": ["c0"]}],
        "harder": [
            {"brainstorming": "...", "chosen_strategy": "s1",
             "updated_question": "q1", "why_harder": "...",
             "verification_criteria": ["c1-a", "c1-b"]},
            {"brainstorming": "...", "chosen_strategy": "s2",
             "updated_question": "q2", "why_harder": "...",
             "verification_criteria": ["c2-a", "c2-b", "c2-c"]},
        ],
        "judge": [
            _judge_payload([{"criterion": "c0", "satisfied": True}]),
            _judge_payload([{"criterion": "c1-a", "satisfied": True},
                            {"criterion": "c1-b", "satisfied": True}]),
            _judge_payload([{"criterion": "c2-a", "satisfied": True},
                            {"criterion": "c2-b", "satisfied": False},
                            {"criterion": "c2-c", "satisfied": True}]),
        ],
    },
    answers_by_attempt={0: "a0", 1: "a1", 2: "a2"},
)
assert r4.final_status == "FAILED_FOUND", r4.final_status
assert r4.attempts[-1].judgment.criteria_failed_count == 1
assert r4.attempts[-1].judgment.verdict == "FAILED"
print("Scenario 4 PASSED")


print("\nAll scenarios passed.")
