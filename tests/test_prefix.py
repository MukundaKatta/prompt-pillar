from prompt_pillar import find_stable_prefix, messages_equal


def _sys(text):
    return {"role": "system", "content": text}


def _user(text):
    return {"role": "user", "content": text}


def test_identical_runs_full_prefix():
    run = [_sys("you are helpful"), _user("hi"), _user("how are you")]
    r = find_stable_prefix([run, list(run), list(run)])
    assert len(r.stable_prefix_messages) == 3
    assert r.first_divergence_index == 3
    assert r.run_count == 3


def test_diverge_at_index_zero_empty_prefix():
    r = find_stable_prefix(
        [
            [_sys("you are helpful v1")],
            [_sys("you are helpful v2")],
        ]
    )
    assert r.stable_prefix_messages == []
    assert r.first_divergence_index == 0
    assert "index 0 differs" in r.divergence_summary


def test_match_for_three_diverge_at_four():
    # First 3 messages (indices 0,1,2) match across both runs; index 3 diverges.
    a = [_sys("s"), _user("u1"), _user("u2"), _user("u3-A")]
    b = [_sys("s"), _user("u1"), _user("u2"), _user("u3-B")]
    r = find_stable_prefix([a, b])
    assert r.first_divergence_index == 3
    assert len(r.stable_prefix_messages) == 3


def test_system_identical_user_differs():
    a = [_sys("s"), _user("hello A")]
    b = [_sys("s"), _user("hello B")]
    r = find_stable_prefix([a, b])
    assert r.first_divergence_index == 1
    assert len(r.stable_prefix_messages) == 1
    assert r.stable_prefix_messages[0]["role"] == "system"


def test_role_only_difference_counts_as_divergence():
    a = [{"role": "system", "content": "hi"}]
    b = [{"role": "user", "content": "hi"}]
    r = find_stable_prefix([a, b])
    assert r.first_divergence_index == 0


def test_dict_key_ordering_does_not_break_equality():
    a = [{"role": "user", "content": "hi", "name": "alice"}]
    b = [{"name": "alice", "content": "hi", "role": "user"}]
    assert messages_equal(a[0], b[0])
    r = find_stable_prefix([a, b])
    assert r.first_divergence_index == 1


def test_empty_runs_input():
    r = find_stable_prefix([])
    assert r.stable_prefix_messages == []
    assert r.run_count == 0


def test_single_run_returns_itself():
    r = find_stable_prefix([[_sys("s"), _user("u")]])
    assert len(r.stable_prefix_messages) == 2
    assert "only one run" in r.divergence_summary


def test_runs_with_different_lengths_full_short_prefix_match():
    a = [_sys("s"), _user("u1")]
    b = [_sys("s"), _user("u1"), _user("u2")]
    r = find_stable_prefix([a, b])
    # Both share the first 2 messages exactly; the second run is longer.
    assert r.first_divergence_index == 2
    assert "different" in r.divergence_summary


def test_four_identical_fifth_has_dict_ordering_difference():
    msg_a = {"role": "user", "content": "x", "name": "alice"}
    msg_b = {"role": "user", "name": "alice", "content": "x"}
    a = [_sys("s"), _user("u1"), _user("u2"), _user("u3"), msg_a]
    b = [_sys("s"), _user("u1"), _user("u2"), _user("u3"), msg_b]
    r = find_stable_prefix([a, b])
    # dict ordering should not cause divergence
    assert r.first_divergence_index == 5
    assert len(r.stable_prefix_messages) == 5


def test_estimated_tokens_grows_with_prefix():
    short = [_sys("s")]
    long = [_sys("s" * 4000)]
    rs = find_stable_prefix([short, short])
    rl = find_stable_prefix([long, long])
    assert rl.estimated_cacheable_tokens > rs.estimated_cacheable_tokens
