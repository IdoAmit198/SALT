from salt_benchmark import tutorial_utils


def test_display_helpers_emit_html_bundle(monkeypatch):
    captured = []

    def fake_display(data, raw=False):
        captured.append((data, raw))

    monkeypatch.setattr(tutorial_utils, "display", fake_display)

    tutorial_utils.show_text("Matrix input", "A=\n1 2")

    data, raw = captured[0]
    assert raw is True
    assert sorted(data) == ["text/html", "text/plain"]
    assert "<h3>Matrix input</h3>" in data["text/html"]
    assert "<pre>A=\n1 2</pre>" in data["text/html"]
    assert data["text/plain"] == "Matrix input\n\nA=\n1 2"


def test_html_table_preserves_multiline_cells():
    html = tutorial_utils._html_table(["Cell"], [["line one\nline two"]])
    assert "line one<br>line two" in html
