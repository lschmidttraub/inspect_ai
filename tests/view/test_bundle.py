import base64
import hashlib
import json
import os
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_helpers.utils import skip_if_trio

from inspect_ai import Task, eval
from inspect_ai._util.error import PrerequisiteError
from inspect_ai._util.file import filesystem
from inspect_ai._view._csp import (
    CSP_FILENAME,
    ContentSecurityPolicyError,
    read_content_security_policy,
)
from inspect_ai._view._dist import resolve_dist_directory
from inspect_ai.dataset import Sample
from inspect_ai.log._bundle import (
    _insert_content_security_policy,
    _prepare_viewer,
    _push_bundle_to_hf,
    bundle_log_dir,
    embed_log_dir,
)
from inspect_ai.scorer import match


@pytest.mark.slow
@skip_if_trio
def test_s3_bundle(mock_s3) -> None:
    # run an eval to generate a log file to this directory
    s3_fs = filesystem("s3://test-bucket/")

    # target directories
    log_dir = "s3://test-bucket/test_s3_bundle/logs"
    output_dir = "s3://test-bucket/test_s3_bundle/view"

    eval(
        tasks=[
            Task(dataset=[Sample(input="Say Hello", target="Hello")], scorer=match())
            for i in range(0, 2)
        ],
        model="mockllm/model",
        log_dir=log_dir,
    )

    # bundle to the output dir
    bundle_log_dir(log_dir, output_dir)

    # ensure files are what we expect
    expected_exact = ["index.html", "assets", "logs", "logs/listing.json"]
    for exp in expected_exact:
        assert s3_fs.exists(os.path.join(output_dir, exp))

    # asset filenames
    assets = s3_fs.ls(os.path.join(output_dir, "assets"))
    asset_names = [os.path.basename(a.name) for a in assets]
    assert "index.js" in asset_names
    assert "index.css" in asset_names


def test_bundle() -> None:
    with tempfile.TemporaryDirectory() as working_dir:
        # run an eval to generate a log file to this directory
        log_dir = os.path.join(working_dir, "logs")
        output_dir = os.path.join(working_dir, "output")

        eval(
            tasks=[
                Task(
                    dataset=[Sample(input="Say Hello", target="Hello")], scorer=match()
                )
                for i in range(0, 2)
            ],
            model="mockllm/model",
            log_dir=log_dir,
        )

        # bundle to the output dir
        bundle_log_dir(log_dir, output_dir)

        # ensure files are what we expect
        expected_exact = ["index.html", "assets", "logs", "logs/listing.json"]
        for exp in expected_exact:
            assert os.path.exists(os.path.join(output_dir, exp))

        # asset filenames
        assert os.path.exists(os.path.join(output_dir, "assets", "index.js"))
        assert os.path.exists(os.path.join(output_dir, "assets", "index.css"))

        # ensure there is a non-listing.json log file present in logs
        non_manifest_logs = [
            f
            for f in os.listdir(os.path.join(output_dir, "logs"))
            if f.endswith(".eval") and f != "logs.eval"
        ]
        assert len(non_manifest_logs) == 2


@skip_if_trio
def test_s3_embed(mock_s3) -> None:
    s3_fs = filesystem("s3://test-bucket/")

    log_dir = "s3://test-bucket/test_s3_embed/logs"

    eval(
        tasks=[
            Task(dataset=[Sample(input="Say Hello", target="Hello")], scorer=match())
            for i in range(0, 2)
        ],
        model="mockllm/model",
        log_dir=log_dir,
    )

    # embed the viewer
    embed_log_dir(log_dir)

    # ensure viewer files are present directly in the log dir
    viewer_expected = ["index.html", "assets", "robots.txt", "listing.json"]
    for exp in viewer_expected:
        assert s3_fs.exists(os.path.join(log_dir, exp))

    # asset filenames
    assets = s3_fs.ls(os.path.join(log_dir, "assets"))
    asset_names = [os.path.basename(a.name) for a in assets]
    assert "index.js" in asset_names
    assert "index.css" in asset_names

    # ensure old viewer/ subdirectory was not created
    assert not s3_fs.exists(os.path.join(log_dir, "viewer"))


def test_embed() -> None:
    with tempfile.TemporaryDirectory() as working_dir:
        log_dir = os.path.join(working_dir, "logs")

        eval(
            tasks=[
                Task(
                    dataset=[Sample(input="Say Hello", target="Hello")], scorer=match()
                )
                for i in range(0, 2)
            ],
            model="mockllm/model",
            log_dir=log_dir,
        )

        # embed the viewer
        embed_log_dir(log_dir)

        # ensure viewer files are present directly in the log dir
        viewer_expected = ["index.html", "assets", "robots.txt", "listing.json"]
        for exp in viewer_expected:
            assert os.path.exists(os.path.join(log_dir, exp))

        # asset filenames
        assert os.path.exists(os.path.join(log_dir, "assets", "index.js"))
        assert os.path.exists(os.path.join(log_dir, "assets", "index.css"))

        # ensure old viewer/ subdirectory was not created
        assert not os.path.exists(os.path.join(log_dir, "viewer"))

        # ensure index.html has the log_dir context set to "."
        with open(os.path.join(log_dir, "index.html")) as f:
            contents = f.read()
        assert '"log_dir": "."' in contents


def test_bundle_output_dir_cannot_be_subdir_of_log_dir(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    output_dir = log_dir / "output"

    log_dir.mkdir()

    with pytest.raises(PrerequisiteError, match="cannot be a subdirectory"):
        bundle_log_dir(str(log_dir), str(output_dir))


def test_bundle_hf_output_dir_allowed(monkeypatch) -> None:
    monkeypatch.setattr(
        "inspect_ai.log._bundle._check_hf_space_exists", lambda repo_id: False
    )
    monkeypatch.setattr(
        "inspect_ai.log._bundle._prepare_viewer", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "inspect_ai.log._bundle.copy_log_files", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "inspect_ai.log._bundle.write_log_listing", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "inspect_ai.log._bundle._push_bundle_to_hf", lambda *args, **kwargs: None
    )

    # Should not raise PrerequisiteError about subdirectory
    bundle_log_dir(log_dir=".", output_dir="hf/username/myspace")


_FIXTURE_INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <script>document.documentElement.dataset.theme = "light";</script>
    <script type="module" src="./assets/index.js"></script>
  </head>
  <body><div id="app"></div></body>
</html>
"""


def _fixture_dist(tmp_path: Path, directives: dict[str, list[str]] | None) -> Path:
    dist_dir = tmp_path / "dist"
    (dist_dir / "assets").mkdir(parents=True)
    (dist_dir / "index.html").write_text(_FIXTURE_INDEX_HTML, encoding="utf-8")
    (dist_dir / "assets" / "index.js").write_text("", encoding="utf-8")
    if directives is not None:
        (dist_dir / CSP_FILENAME).write_text(
            json.dumps({"version": 1, "directives": directives}), encoding="utf-8"
        )
    return dist_dir


class _HeadChildren(HTMLParser):
    """Collect the start tags (with decoded attributes) directly inside <head>."""

    def __init__(self) -> None:
        super().__init__()
        self.children: list[tuple[str, dict[str, str | None]]] = []
        self._depth: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "head":
            self._depth = 0
        elif self._depth is not None:
            if self._depth == 0:
                self.children.append((tag, dict(attrs)))
            if tag not in ("meta", "link"):
                self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self._depth = None
        elif self._depth:
            self._depth -= 1


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dist_dir: Path) -> str:
    monkeypatch.setattr("inspect_ai.log._bundle._dist_dir", lambda: dist_dir.as_posix())
    working_dir = tmp_path / "bundle"
    working_dir.mkdir()
    _prepare_viewer(str(working_dir), log_dir="logs", abs_log_dir="/abs/logs")
    return (working_dir / "index.html").read_text(encoding="utf-8")


def test_bundle_inserts_viewer_csp_meta_first_in_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist_dir = _fixture_dist(
        tmp_path,
        {
            "default-src": ["'none'"],
            "script-src": ["'self'", "'sha256-abc+/='"],
            "img-src": ['https://cdn.example/a?b&c<"d">'],
        },
    )
    index_html = _prepare(tmp_path, monkeypatch, dist_dir)

    parser = _HeadChildren()
    parser.feed(index_html)
    first_tag, first_attrs = parser.children[0]
    assert first_tag == "meta"
    assert first_attrs == {
        "http-equiv": "Content-Security-Policy",
        "content": (
            "default-src 'none'; script-src 'self' 'sha256-abc+/='; "
            'img-src https://cdn.example/a?b&c<"d">'
        ),
    }
    assert "frame-ancestors" not in index_html
    assert "&amp;c&lt;&quot;d&quot;&gt;" in index_html
    assert "default-src 'none'; script-src 'self'" in index_html

    # the log_dir_context data block is still injected, after the inline scripts
    tags = [tag for tag, _ in parser.children]
    assert tags == ["meta", "meta", "script", "script", "script"]
    assert parser.children[-1][1] == {
        "id": "log_dir_context",
        "type": "application/json",
    }
    assert '{"log_dir": "logs", "abs_log_dir": "/abs/logs"}' in index_html


def test_bundle_without_viewer_csp_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_html = _prepare(tmp_path, monkeypatch, _fixture_dist(tmp_path, None))

    context = '{"log_dir": "logs", "abs_log_dir": "/abs/logs"}'
    assert index_html == _FIXTURE_INDEX_HTML.replace(
        "</head>",
        f'  <script id="log_dir_context" type="application/json">{context}</script>\n  </head>',
    )


def _bundle_to(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_dir: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run `bundle_log_dir` over the fixture dist, capturing the HF push."""
    dist_dir = _fixture_dist(
        tmp_path,
        {
            "default-src": ["'none'"],
            "font-src": ["'self'"],
            "connect-src": ["'self'"],
        },
    )
    monkeypatch.setattr("inspect_ai.log._bundle._dist_dir", lambda: dist_dir.as_posix())
    monkeypatch.setattr(
        "inspect_ai.log._bundle._check_hf_space_exists", lambda repo_id: False
    )
    captured: dict[str, Any] = {}

    def copy_log_files(*args: Any) -> None:
        captured["fs_options"] = args[3]

    def push(working_dir: str, output_dir: str, private: bool | None) -> None:
        captured["index_html"] = (Path(working_dir) / "index.html").read_text()
        captured["private"] = private

    monkeypatch.setattr("inspect_ai.log._bundle.copy_log_files", copy_log_files)
    monkeypatch.setattr(
        "inspect_ai.log._bundle.write_log_listing", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("inspect_ai.log._bundle._push_bundle_to_hf", push)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    bundle_log_dir(log_dir=str(log_dir), output_dir=output_dir, **kwargs)
    return captured


def test_hf_bundle_allows_hf_cdn_redirects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _bundle_to(tmp_path, monkeypatch, "hf/username/myspace")
    index_html = captured["index_html"]
    assert "connect-src 'self' https://huggingface.co https://*.hf.co" in index_html
    assert "font-src 'self' https://huggingface.co https://*.hf.co" in index_html
    assert '"allow_log_redirects": true' in index_html
    assert captured["private"] is None


def test_local_bundle_keeps_connect_src_self(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    _bundle_to(tmp_path, monkeypatch, str(output_dir))
    index_html = (output_dir / "index.html").read_text()
    assert "connect-src 'self'\"" in index_html
    assert "hf.co" not in index_html
    assert "allow_log_redirects" not in index_html


def test_hf_bundle_accepts_legacy_private_fs_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _bundle_to(
        tmp_path,
        monkeypatch,
        "hf/username/myspace",
        fs_options={"private": True, "anon": True},
    )
    assert captured["private"] is True
    # `private` is a bundle option, not a filesystem argument
    assert captured["fs_options"] == {"anon": True}


class _FakeHfApi:
    """Records `HfApi` calls; `existing` maps repo ids to their privacy."""

    def __init__(self, existing: dict[str, bool]) -> None:
        self.existing = existing
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self) -> "_FakeHfApi":
        return self

    def repo_info(self, repo_id: str, repo_type: str) -> SimpleNamespace:
        if repo_id not in self.existing:
            raise LookupError(repo_id)
        return SimpleNamespace(private=self.existing[repo_id])

    def create_repo(self, **kwargs: Any) -> None:
        self.calls.append(("create_repo", kwargs))
        self.existing[kwargs["repo_id"]] = kwargs["private"]

    def update_repo_settings(self, **kwargs: Any) -> None:
        self.calls.append(("update_repo_settings", kwargs))
        self.existing[kwargs["repo_id"]] = kwargs["private"]

    def upload_folder(self, **kwargs: Any) -> None:
        self.calls.append(("upload_folder", kwargs))


def _push_with_fake_hf(
    monkeypatch: pytest.MonkeyPatch,
    existing: dict[str, bool],
    private: bool | None,
) -> _FakeHfApi:
    api = _FakeHfApi(existing)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=api))
    _push_bundle_to_hf("/bundle", "hf/user/space", private)
    return api


@pytest.mark.parametrize("private", [None, False, True])
def test_push_bundle_to_hf_creates_public_space_by_default(
    monkeypatch: pytest.MonkeyPatch, private: bool | None
) -> None:
    api = _push_with_fake_hf(monkeypatch, {}, private)
    assert api.calls[0] == (
        "create_repo",
        {
            "repo_id": "user/space",
            "repo_type": "space",
            "space_sdk": "static",
            "private": bool(private),
        },
    )
    assert api.calls[1][0] == "upload_folder"


def test_push_bundle_to_hf_keeps_existing_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _push_with_fake_hf(monkeypatch, {"user/space": True}, None)
    assert [name for name, _ in api.calls] == ["upload_folder"]
    assert api.existing["user/space"] is True


def test_push_bundle_to_hf_updates_existing_visibility_when_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _push_with_fake_hf(monkeypatch, {"user/space": True}, False)
    assert api.calls[0] == (
        "update_repo_settings",
        {"repo_id": "user/space", "repo_type": "space", "private": False},
    )
    assert api.existing["user/space"] is False


def test_csp_extra_sources_require_the_directive(tmp_path: Path) -> None:
    dist_dir = _fixture_dist(tmp_path, {"default-src": ["'none'"]})
    with pytest.raises(ContentSecurityPolicyError, match="'connect-src'"):
        read_content_security_policy(dist_dir, {"connect-src": ["https://*.hf.co"]})


def test_bundle_viewer_csp_requires_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist_dir = _fixture_dist(tmp_path, {"default-src": ["'none'"]})
    (dist_dir / "index.html").write_text("<html><body></body></html>")
    with pytest.raises(RuntimeError, match="no <head> element"):
        _prepare(tmp_path, monkeypatch, dist_dir)


# Script types the browser executes (and so CSP governs); anything else, like
# `application/json`, is an inert data block.
_EXECUTABLE_SCRIPT_TYPES = frozenset({"", "module", "text/javascript"})


class _InlineScripts(HTMLParser):
    """Collect the exact text of each executable inline <script>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.scripts: list[str] = []
        self._current: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "script":
            return
        attributes = dict(attrs)
        script_type = (attributes.get("type") or "").strip().lower()
        if "src" not in attributes and script_type in _EXECUTABLE_SCRIPT_TYPES:
            self._current = []

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._current.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._current is not None:
            self.scripts.append("".join(self._current))
            self._current = None


def test_shipped_viewer_csp_matches_its_index_html() -> None:
    dist_dir = resolve_dist_directory()
    policy = read_content_security_policy(dist_dir)
    if policy is None:
        pytest.skip(f"viewer dist has no {CSP_FILENAME}")
    index_html = (dist_dir / "index.html").read_text(encoding="utf-8")

    script_src = next(
        directive.split()[1:]
        for directive in policy.split("; ")
        if directive.split()[0] == "script-src"
    )
    parser = _InlineScripts()
    parser.feed(index_html)
    assert parser.scripts, "expected the viewer's inline theme bootstrap script"
    for script in parser.scripts:
        digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest())
        assert f"'sha256-{digest.decode('ascii')}'" in script_src, script[:80]

    bundled = _insert_content_security_policy(index_html, policy).encode("utf-8")
    charset = bundled.find(b"<meta charset")
    assert charset != -1
    assert bundled.index(b">", charset) < 1024
