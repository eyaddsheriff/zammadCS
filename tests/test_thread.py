"""HTML stripping and thread flattening.

The internal-note test is the important one here: a regression there leaks
agent-only notes into a customer-facing reply.
"""

from zammad_agent.pipeline.thread import article_text, flatten_thread, html_to_text


def _article(body, sender="Customer", internal=False, content_type="text/plain"):
    return {
        "body": body,
        "sender": sender,
        "internal": internal,
        "content_type": content_type,
    }


def test_strips_style_blocks():
    # Email HTML routinely carries <style> blocks that would otherwise reach
    # the prompt as hundreds of tokens of CSS.
    assert "color" not in html_to_text("<style>.a{color:red}</style><p>Hello</p>")


def test_decodes_entities():
    assert html_to_text("<p>Tom &amp; Jerry&#39;s</p>") == "Tom & Jerry's"


def test_block_tags_separate_content():
    # Without this every paragraph runs together and the message loses shape.
    # A blank line between blocks costs one token and buys the model clearer
    # structure, which is a trade worth making.
    assert html_to_text("<div>One</div><div>Two</div>") == "One\n\nTwo"


def test_nesting_cannot_produce_runaway_blank_lines():
    # Email HTML nests divs many levels deep. Each level emits newlines, so
    # without the collapse a short message could arrive as pages of whitespace.
    assert html_to_text("<div><div><div><p>One</p></div></div></div><p>Two</p>") == "One\n\nTwo"


def test_collapses_layout_whitespace():
    assert html_to_text("<p>too    many     spaces</p>") == "too many spaces"


def test_survives_unbalanced_tags():
    # Real email HTML is frequently malformed. A stray closing tag must not
    # suppress the rest of the body.
    assert "visible" in html_to_text("</style><p>visible</p>")


def test_plain_text_body_is_left_alone():
    assert article_text(_article("a < b and c > d")) == "a < b and c > d"


def test_html_body_is_converted():
    assert article_text(_article("<p>hi</p>", content_type="text/html")) == "hi"


def test_labels_each_turn_with_sender():
    thread = flatten_thread([_article("Help"), _article("Sure", sender="Agent")])
    assert thread == "Customer: Help\n\nAgent: Sure"


def test_internal_notes_excluded_by_default():
    articles = [_article("Public"), _article("Do not tell them", sender="Agent", internal=True)]
    assert "Do not tell them" not in flatten_thread(articles)


def test_internal_notes_included_on_request():
    articles = [_article("Public"), _article("Do not tell them", sender="Agent", internal=True)]
    assert "Do not tell them" in flatten_thread(articles, include_internal=True)


def test_empty_articles_are_dropped():
    # Zammad articles can carry an empty body; an unlabelled "Customer:" line
    # would just waste prompt tokens.
    assert flatten_thread([_article(""), _article(None), _article("real")]) == "Customer: real"


def test_truncation_keeps_the_latest_message():
    # The customer's most recent message defines the current ask, so the old
    # end is what gets dropped.
    articles = [_article("x" * 200), _article("the newest thing", sender="Agent")]
    thread = flatten_thread(articles, max_chars=40)
    assert "the newest thing" in thread
    assert thread.startswith("[earlier messages truncated]")


def test_no_truncation_when_within_budget():
    assert not flatten_thread([_article("short")], max_chars=1000).startswith("[earlier")
