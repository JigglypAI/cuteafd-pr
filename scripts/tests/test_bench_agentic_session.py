import importlib.util
import io
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "bench/bench-agentic-session.py"


def load():
    spec = importlib.util.spec_from_file_location("bench_agentic_session", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sse(events):
    return io.BytesIO(b"".join(b"data: " + json.dumps(e).encode() + b"\n" for e in events) + b"data: [DONE]\n")


def test_fixture_fails_exactly_the_planted_tests_and_the_fixes_pass():
    bench = load()
    workspace = bench.Workspace()
    assert 40 <= len(workspace.files) <= 60
    failing = sorted(line.split()[1] for line in workspace.run_tests().splitlines() if line.startswith("FAILED"))
    assert failing == ["tests/test_dates.py::test_december_month_end", "tests/test_money.py::test_negative_rounding",
                       "tests/test_parser.py::test_doubled_quotes", "tests/test_parser.py::test_quoted_commas",
                       "tests/test_rates.py::test_expired_rate_refetched", "tests/test_report.py::test_december_report"]
    assert not any(workspace.passing(task["tests"]) for task in bench.TASKS)
    workspace.edit("ledger/money.py", "        return -quotient - (1 if remainder else 0)", "        return -quotient")
    workspace.edit("ledger/rates.py", "now - cached.fetched_at > self.ttl", "now - cached.fetched_at <= self.ttl")
    workspace.edit("ledger/dates.py", "date(day.year, day.month % 12 + 1, 1)",
                   "date(day.year + day.month // 12, day.month % 12 + 1, 1)")
    assert [workspace.passing(task["tests"]) for task in bench.TASKS] == [True, False, True, True]
    # The run is deterministic: no timings, no temporary paths.
    assert workspace.run_tests("tests/test_money.py") == workspace.run_tests("tests/test_money.py")
    assert "<repo>" not in workspace.run_tests() and "tmp" not in workspace.run_tests()


def test_tools_are_deterministic_and_validated():
    bench = load()
    workspace = bench.Workspace()
    text, problem = workspace.execute("read_file", json.dumps({"path": "ledger/dates.py", "start_line": 1, "end_line": 3}))
    assert problem is None and text.splitlines()[1] == '    1\t"""Calendar helpers for monthly reporting periods."""'
    assert "[lines 4-" in text
    assert workspace.execute("list_dir", '{"path": "."}')[0].splitlines()[:3] == ["CHANGELOG.md", "CONTRIBUTING.md", "README.md"]
    assert "ledger/rates.py:" in workspace.execute("grep", '{"pattern": "class RateCache"}')[0]
    assert workspace.execute("edit", json.dumps({"path": "ledger/tags.py", "old_string": "import", "new_string": "x"}))[0] \
        .startswith("error: old_string is not unique")
    assert workspace.execute("read_file", '{"path": "../etc/passwd"}')[0].startswith("error: path")
    assert workspace.execute("write_file", '{"path": "notes.txt", "content": "hi"}')[0] == "wrote 2 bytes to notes.txt"
    # Invalid calls are answered with an error and counted as invalid.
    assert workspace.execute("read_file", "{not json")[1].startswith("arguments are not JSON")
    assert workspace.execute("read_file", '{"file": "x"}')[1].startswith("arguments do not match")
    assert workspace.execute("delete_everything", "{}")[1] == "unknown tool 'delete_everything'"


def test_missing_jsonschema_keeps_the_validation_error_contract(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    bench = load()
    assert bench.validate_call("read_file", '{"file": "x"}').startswith("arguments do not match the schema:")
    assert bench.validate_call("read_file", '{"path": "x"}') is None


def test_stream_accumulates_tool_call_deltas_and_times_first_output(monkeypatch):
    bench = load()
    events = [
        {"choices": [{"delta": {"reasoning_content": "Read the test first."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c0", "function": {"name": "read_", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": '{"path": '}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "c1", "function": {"name": "run_tests", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"tests/test_money.py"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 9000, "prompt_cache_hit_tokens": 8500, "completion_tokens": 41}},
    ]
    monkeypatch.setattr(bench, "open_request", lambda *a, **k: sse(events))
    ticks = iter([0, 2, 3, 4, 5, 6, 12, 13, 14])
    result = bench.stream_turn("unused", {}, clock=lambda: next(ticks))
    assert result["calls"] == [{"id": "c0", "name": "read_file", "arguments": '{"path": "tests/test_money.py"}'},
                               {"id": "c1", "name": "run_tests", "arguments": "{}"}]
    assert (result["first_output_seconds"], result["finish_seconds"], result["done"]) == (2, 12, True)
    previous = {"prompt_tokens": 8000, "completion_tokens": 501}
    metrics = bench.turn_metrics(result, previous, 1, 12.5)
    assert metrics["decode_tps"] == 40 / 10
    assert (metrics["tool_calls"], metrics["invalid_tool_calls"], metrics["prefill_tokens"]) == (2, 0, 500)
    assert (metrics["prompt_reused"], metrics["full_turn_reused"]) == (True, True)
    previous["completion_tokens"] = 502
    assert bench.turn_metrics(result, previous, 1, 1)["full_turn_reused"] is False


def scripted_server(bench, monkeypatch, replies):
    """A fake API: returns scripted replies and records every request body."""
    bodies = []

    def fake(base, body, api_key=None, clock=None):
        bodies.append(json.loads(json.dumps(body)))
        reply = replies[min(len(bodies), len(replies)) - 1]
        prompt = sum(len(json.dumps(m)) for m in body["messages"]) // 4
        return dict(reasoning=reply.get("reasoning", "thinking"), content=reply.get("content", ""),
                    calls=reply.get("calls", []), finish_reason="tool_calls" if reply.get("calls") else "stop",
                    first_output_seconds=0.5, first_content_seconds=None, finish_seconds=1.5, done=True,
                    usage={"prompt_tokens": prompt, "prompt_cache_hit_tokens": 0, "completion_tokens": 20})

    monkeypatch.setattr(bench, "stream_turn", fake)
    return bodies


def test_record_then_replay_sends_identical_prompts_per_repeat(tmp_path, monkeypatch):
    bench = load()
    fix = {"path": "ledger/money.py", "old_string": "        return -quotient - (1 if remainder else 0)",
           "new_string": "        return -quotient"}
    replies = [
        {"calls": [{"id": "a", "name": "read_file", "arguments": '{"path": "tests/test_money.py"}'},
                   {"id": "", "name": "grep", "arguments": '{"pattern": "def round_half_even"}'}]},
        {"calls": [{"id": "b", "name": "edit", "arguments": json.dumps(fix)}]},
        {"calls": [{"id": "c", "name": "run_tests", "arguments": '{"path": "tests/test_money.py"}'}]},
        {"content": "Fixed the negative rounding."},
    ]
    bodies = scripted_server(bench, monkeypatch, replies)
    recording = tmp_path / "record.json"
    assert bench.main(["record", "--output", str(recording), "--seeds", "0", "--label", "t"]) == 0
    report = json.loads(recording.read_text())
    session = report["sessions"][0]
    assert (session["task"], session["success"], session["final_answer"]) == ("money", True, True)
    assert [t["messages"] for t in session["turns"]] == [2, 5, 7, 9]
    assert [m["role"] for m in session["messages"][2:5]] == ["assistant", "tool", "tool"]
    assert session["messages"][3]["tool_call_id"] == "a" and session["messages"][4]["tool_call_id"] == "call_0_1"
    assert session["messages"][2]["reasoning_content"] == "thinking"
    assert "{{NONCE}}" in session["messages"][0]["content"]
    assert bodies[0]["tools"] == bench.TOOLS and bodies[0]["temperature"] == 0 and bodies[0]["max_tokens"] == 8192
    assert bodies[0]["reasoning_effort"] == "high" and bodies[0]["messages"][0]["content"].startswith("session ")

    bodies.clear()
    replay = lambda label, output: bench.main(["replay", "--recording", str(recording), "--output", str(tmp_path / output),
                                               "--label", label, "--repeats", "2", "--concurrency", "1", "4",
                                               "--stagger", "0"])
    assert replay("ab", "a.json") == 0
    first = [json.dumps(b["messages"]) for b in bodies]
    bodies.clear()
    assert replay("ab", "b.json") == 0
    assert [json.dumps(b["messages"]) for b in bodies] == first, "both arms see identical prompts"
    # Four turns per run; runs (repeat x concurrency) never share a nonce.
    systems = [b["messages"][0]["content"].splitlines()[0] for b in bodies]
    assert len(set(systems)) == 4 and all(systems.count(s) == 4 for s in set(systems))
    assert json.dumps(bodies[3]["messages"][1:]) == json.dumps(session["messages"][1:9])
    summary = json.loads((tmp_path / "a.json").read_text())["summary"]
    assert set(summary) == {"c1", "c4"} and summary["c1"]["turns"] == 8 and summary["c1"]["tool_call_validity"] == 1.0
    assert "ttft_later_s" in bench.render(summary)


def test_echo_reasoning_off_keeps_reasoning_out_of_the_history(tmp_path, monkeypatch):
    bench = load()
    bodies = scripted_server(bench, monkeypatch, [{"calls": [{"id": "a", "name": "list_dir", "arguments": '{"path": "."}'}]},
                                                  {"content": "done"}])
    bench.main(["record", "--output", str(tmp_path / "r.json"), "--seeds", "2", "--echo-reasoning", "off"])
    assert "reasoning_content" not in bodies[1]["messages"][2]
    assert json.loads((tmp_path / "r.json").read_text())["sessions"][0]["success"] is False


@pytest.mark.parametrize("identical", [False, True])
def test_bench_ab_concurrency_distinct_default_and_explicit_opt_out(tmp_path, monkeypatch, identical):
    spec = importlib.util.spec_from_file_location("bench_ab", SCRIPT.with_name("bench-ab.py"))
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    monkeypatch.setattr(ab, "REPO", tmp_path)
    monkeypatch.setattr(ab, "stop_all", lambda *a: None)
    commands = []

    def fake_run(cmd, cwd=None, log=None, timeout=0):
        commands.append(cmd)
        if "--output" in cmd:
            output = Path(cmd[cmd.index("--output") + 1])
            output.write_text(json.dumps({"summaries": [{"concurrency": 16, "median_aggregate_tps": 50}],
                                          "median_weighted_observed_decode_tokens_per_second": 20}))

    monkeypatch.setattr(ab, "run", fake_run)
    monkeypatch.setattr(ab.sys, "argv", ["bench-ab.py", "--label", "test", "--arm", "a=" + str(tmp_path),
                                       "--arm", "b=" + str(tmp_path), "--layouts", "1", "--sessions", "2",
                                       "--concurrency", "16", "--nonce-seed", "7",
                                       *(["--identical-prompts"] if identical else [])])
    ab.main()
    concurrent = [c for c in commands if any(x.endswith("bench-concurrent-api.py") for x in c)]
    assert len(concurrent) == 2
    for command in concurrent:
        assert ("--distinct-prompts" in command) is not identical
        assert "--identical-prompts" not in command
        assert command[command.index("--nonce") + 1] == "ab-7"
        assert command[command.index("--prompt-label") + 1] == "ab"
    assert [c[c.index("--label") + 1] for c in concurrent] == ["a", "b"]


@pytest.mark.parametrize("task_name", ["task", "backoff"])
def test_bench_ab_agentic_battery_rows_and_summary(tmp_path, monkeypatch, task_name):
    tmp_path = tmp_path / task_name
    tmp_path.mkdir()
    spec = importlib.util.spec_from_file_location("bench_ab", SCRIPT.with_name("bench-ab.py"))
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    commands = []

    def fake_run(cmd, cwd=None, log=None, timeout=0):
        commands.append(cmd)
        output = Path(cmd[cmd.index("--output") + 1])
        value = 2.0 if output.parent.name == "off" else 0.5
        output.write_text(json.dumps({"summary": {"c1": {"ttft_later_s": value, "decode_tps": 30.0},
                                                  "c4": {"ttft_later_s": 2 * value, "decode_tps": 90.0}}}))

    monkeypatch.setattr(ab, "run", fake_run)
    args = type("Args", (), dict(agentic_recording=tmp_path / "rec.json", base_url="http://x", concurrency=[1, 4],
                                 agentic_arg=["--model=m"], nonce_seed=7, repeats=3))()
    rows = []
    for arm in ("off", "on"):
        out = tmp_path / arm
        out.mkdir()
        rows.append(ab.agentic_session(arm, 1, 12.0, out, out / "log", "python", args))
    assert commands[0][commands[0].index("--label") + 1] == "ab-7-warmup"
    assert commands[1][commands[1].index("--repeats") + 1] == "3"
    assert rows[0]["c1_ttft_later_s"] == 2.0 and rows[1]["c4_ttft_later_s"] == 1.0
    table = ab.summarize(rows, ["off", "on"])
    line = next(l for l in table.splitlines() if l.startswith("c1_ttft_later_s"))
    assert line.split()[-1] == "0.250"


def chain_server(bench, monkeypatch, replies, prompt_step=1000):
    """A fake API whose prompt grows by `prompt_step` tokens per request."""
    bodies = []

    def fake(base, body, api_key=None, clock=None):
        bodies.append(json.loads(json.dumps(body)))
        reply = replies[(len(bodies) - 1) % len(replies)]
        return dict(reasoning=reply.get("reasoning", "thinking"), content=reply.get("content", ""),
                    calls=reply.get("calls", []), finish_reason="tool_calls" if reply.get("calls") else "stop",
                    first_output_seconds=0.5, first_content_seconds=None, finish_seconds=1.5, done=True,
                    usage={"prompt_tokens": prompt_step * len(bodies), "prompt_cache_hit_tokens": 0,
                           "completion_tokens": 20})

    monkeypatch.setattr(bench, "stream_turn", fake)
    return bodies


def test_chain_tasks_appends_the_next_task_and_stops_at_the_target(tmp_path, monkeypatch):
    bench = load()
    replies = [{"calls": [{"id": "a", "name": "list_dir", "arguments": '{"path": "."}'}]}, {"content": "done"}]
    bodies = chain_server(bench, monkeypatch, replies)
    out = tmp_path / "chain.json"
    assert bench.main(["record", "--output", str(out), "--seeds", "1", "--chain-tasks", "--target-context", "9500",
                       "--max-turns", "50", "--label", "c"]) == 0
    session = json.loads(out.read_text())["sessions"][0]
    # Ten requests: the tenth prompt (10000) reaches the target before its completion is added.
    assert len(bodies) == len(session["turns"]) == 10 and session["stop"] == "target_context"
    names = [t["task"] for t in session["chain"]]
    # Seed 1 starts at the second task and cycles; five tasks were opened in ten turns.
    assert names == ["parser", "rates", "dates", "money", "parser"]
    assert session["chain"][4]["cycle"] == 1 and session["chain"][0]["first_turn"] == 0
    users = [m["content"] for m in session["messages"] if m["role"] == "user"]
    assert len(users) == 5 and users[1].startswith("Thanks, that one is done.") and bench.TASKS[2]["prompt"] in users[1]
    assert "reset to its original state" in users[4] and "reset" not in users[3]
    # One history: every request extends the previous one, reasoning echoed back.
    for before, after in zip(bodies, bodies[1:]):
        assert after["messages"][:len(before["messages"])][1:] == before["messages"][1:]
    assert bodies[-1]["messages"][2]["reasoning_content"] == "thinking"
    assert [t["task_index"] for t in session["turns"]] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert all(t["success"] is False for t in session["chain"][:4])


def test_chain_tasks_respects_max_turns_and_fresh_seeds(tmp_path, monkeypatch):
    bench = load()
    bodies = chain_server(bench, monkeypatch, [{"calls": [{"id": "a", "name": "list_dir", "arguments": '{"path": "."}'}]}])
    out = tmp_path / "chain.json"
    assert bench.main(["record", "--output", str(out), "--seeds", "0", "2", "--chain-tasks",
                       "--target-context", "1000000", "--max-turns", "3", "--label", "c"]) == 0
    sessions = json.loads(out.read_text())["sessions"]
    assert [s["stop"] for s in sessions] == ["max_turns", "max_turns"] and len(bodies) == 6
    assert [s["chain"][0]["task"] for s in sessions] == ["money", "rates"]
    nonces = {b["messages"][0]["content"].splitlines()[0] for b in bodies}
    assert len(nonces) == 2, "each seed's session has its own nonce"


def test_chain_tasks_argument_pairing_and_key_file(tmp_path, monkeypatch):
    bench = load()
    chain_server(bench, monkeypatch, [{"content": "done"}])
    with pytest.raises(SystemExit):
        bench.main(["record", "--output", str(tmp_path / "a.json"), "--chain-tasks"])
    with pytest.raises(SystemExit):
        bench.main(["record", "--output", str(tmp_path / "a.json"), "--target-context", "4096"])
    with pytest.raises(SystemExit):
        bench.main(["record", "--output", str(tmp_path / "a.json"), "--chain-tasks", "--target-context", "10"])
    key = tmp_path / "key"
    key.write_text("secret\n")
    seen = []
    monkeypatch.setattr(bench, "stream_turn", lambda base, body, api_key=None, clock=None: seen.append((base, api_key))
                        or dict(reasoning="", content="ok", calls=[], finish_reason="stop", first_output_seconds=0.1,
                                first_content_seconds=0.1, finish_seconds=0.2, done=True,
                                usage={"prompt_tokens": 2000, "prompt_cache_hit_tokens": 0, "completion_tokens": 1}))
    monkeypatch.setenv("CUTEAFD_SMOKE_URL", "http://card:9")
    monkeypatch.setenv("CUTEAFD_API_KEY_FILE", str(key))
    assert bench.main(["record", "--output", str(tmp_path / "b.json"), "--seeds", "0", "--chain-tasks",
                       "--target-context", "2000"]) == 0
    assert seen == [("http://card:9", "secret")]


def test_chain_tasks_saves_every_turn_and_honours_the_deadline(tmp_path, monkeypatch):
    bench = load()
    bodies = chain_server(bench, monkeypatch, [{"calls": [{"id": "a", "name": "list_dir", "arguments": '{"path": "."}'}]}])
    clock = iter([0.0, 1.0, 2.0, 100.0, 101.0])
    monkeypatch.setattr(bench.time, "monotonic", lambda: next(clock))
    out = tmp_path / "chain.json"
    seen = []
    original = out.write_text
    assert bench.main(["record", "--output", str(out), "--seeds", "0", "--chain-tasks", "--target-context", "1000000",
                       "--max-turns", "50", "--deadline-s", "50"]) == 0
    session = json.loads(out.read_text())["sessions"][0]
    assert session["stop"] == "deadline" and len(bodies) == len(session["turns"]) == 2
