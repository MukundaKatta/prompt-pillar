from prompt_pillar import Pillar


def _sys(text):
    return {"role": "system", "content": text}


def _user(text):
    return {"role": "user", "content": text}


def test_pillar_identical_runs_pass():
    big_sys = _sys("you are helpful" * 500)
    runs = [[big_sys, _user("u")] for _ in range(3)]
    p = Pillar(threshold_tokens=10)
    rep = p.analyze(runs)
    assert rep.meets_threshold is True
    assert "all runs share" in rep.reason or "all runs identical" in rep.reason


def test_pillar_timestamp_injection_detected():
    a = [_sys("you are helpful. now: 2026-05-24T10:34"), _user("hi")]
    b = [_sys("you are helpful. now: 2026-05-24T10:35"), _user("hi")]
    rep = Pillar(threshold_tokens=10).analyze([a, b])
    assert "timestamp" in rep.reason.lower()
    assert "user-turn" in rep.recommendation or "user turn" in rep.recommendation


def test_pillar_uuid_injection_detected():
    u1 = "request-id 11111111-2222-3333-4444-555555555555"
    u2 = "request-id 66666666-7777-8888-9999-aaaaaaaaaaaa"
    a = [_sys(u1), _user("hi")]
    b = [_sys(u2), _user("hi")]
    rep = Pillar(threshold_tokens=10).analyze([a, b])
    assert "uuid" in rep.reason.lower() or "per-call" in rep.reason.lower()


def test_pillar_role_change_diagnosis():
    a = [{"role": "system", "content": "x"}]
    b = [{"role": "user", "content": "x"}]
    rep = Pillar(threshold_tokens=10).analyze([a, b])
    assert "role" in rep.reason.lower()


def test_pillar_below_threshold_flagged():
    # 1-char system msg, threshold 10000 tokens; prefix is far below.
    a = [_sys("x")]
    b = [_sys("x")]
    rep = Pillar(threshold_tokens=10000).analyze([a, b])
    assert rep.meets_threshold is False
    assert "BELOW" in str(rep)


def test_pillar_single_run_recommendation():
    a = [_sys("only one run here")]
    rep = Pillar(threshold_tokens=10).analyze([a])
    assert "2 or more" in rep.recommendation or "2 runs" in rep.recommendation.lower()


def test_pillar_str_contains_summary_lines():
    a = [_sys("y" * 8000)]
    b = [_sys("y" * 8000)]
    s = str(Pillar(threshold_tokens=10).analyze([a, b]))
    assert "Stable prefix" in s
    assert "Reason" in s
    assert "Recommendation" in s
