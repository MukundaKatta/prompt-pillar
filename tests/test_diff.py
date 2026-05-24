from prompt_pillar import diff_messages


def test_identical_messages_report_identical():
    a = {"role": "user", "content": "hello world"}
    b = {"role": "user", "content": "hello world"}
    r = diff_messages(a, b)
    assert r.identical is True
    assert r.role_changed is False
    assert r.content_unified_diff == ""


def test_role_change_flagged():
    a = {"role": "system", "content": "x"}
    b = {"role": "user", "content": "x"}
    r = diff_messages(a, b)
    assert r.role_changed is True
    assert r.role_a == "system"
    assert r.role_b == "user"


def test_content_diff_first_char_diverge():
    a = {"role": "user", "content": "hello A world"}
    b = {"role": "user", "content": "hello B world"}
    r = diff_messages(a, b)
    assert r.identical is False
    assert r.first_char_diverge == 6  # position of "A" vs "B"
    assert "hello A world" in r.content_unified_diff
    assert "hello B world" in r.content_unified_diff


def test_structured_content_renders_for_diff():
    a = {"role": "user", "content": [{"type": "text", "text": "a"}]}
    b = {"role": "user", "content": [{"type": "text", "text": "b"}]}
    r = diff_messages(a, b)
    assert r.identical is False
    assert "a" in r.content_unified_diff and "b" in r.content_unified_diff
