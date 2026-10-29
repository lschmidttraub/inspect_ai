"""Saved sample-list views for `inspect view`, kept per log folder.

The viewer saves named views (filter, columns, sort, layout) for the folder
holding the log being viewed. They are stored here, in the user's inspect data
dir, so they outlast server restarts and don't depend on the browser's
per-origin storage (a different port or host starts empty). The server treats
each view as opaque JSON apart from its name; the viewer validates the rest.

File format::

    {"version": 1,
     "folders": {"/abs/log/folder": [view, ...]},
     "legacy_views_folder": "/abs/log/folder"}

`legacy_views_folder` (optional, set by hand) is where the viewer files views
it saved before views were per folder; without it they go to the first folder
opened.
"""

import json
import os
import time
import urllib.parse
from logging import getLogger
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from inspect_ai._util.appdirs import inspect_data_dir

from .common import normalize_uri

logger = getLogger(__name__)

SAVED_VIEWS_VERSION = 1
MAX_SAVED_VIEWS_BYTES = 5 * 1024 * 1024


class SavedViews(BaseModel):
    views: list[dict[str, Any]]
    """Saved views, in display order."""


class SavedViewsResponse(SavedViews):
    folder: str
    """The folder the views belong to, as stored (see `folder_key`)."""

    path: str
    """File the views are stored in."""

    legacy_views_folder: str | None = None
    """Where to file views saved before views were per folder."""


def default_saved_views_file() -> Path:
    """The saved-views file in the user's inspect data dir."""
    return inspect_data_dir("view") / "saved_views.json"


def folder_key(folder: str) -> str:
    """One spelling per folder: `file:///a/b/`, `/a/b/` and `/a/b` all map to `/a/b`."""
    uri = normalize_uri(folder)
    if uri.startswith("file://"):
        uri = urllib.parse.urlparse(uri).path
    return uri.rstrip("/") or "/"


def _read_file(path: Path) -> dict[str, Any]:
    """The whole file, or an empty one if it doesn't exist yet.

    A file that can't be parsed is renamed aside (`<name>.corrupt-<time>`)
    and logged rather than overwritten by the next save.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"version": SAVED_VIEWS_VERSION, "folders": {}}
    try:
        data = json.loads(text)
        folders = data["folders"]
        if not isinstance(folders, dict) or not all(
            isinstance(views, list) and all(isinstance(v, dict) for v in views)
            for views in folders.values()
        ):
            raise ValueError("'folders' must map folders to lists of views")
        legacy = data.get("legacy_views_folder")
        if legacy is not None and not isinstance(legacy, str):
            raise ValueError("'legacy_views_folder' must be a string")
        return dict(data)
    except (ValueError, KeyError, TypeError) as ex:
        aside = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
        os.replace(path, aside)
        logger.warning(f"Moved unreadable saved views file {path} to {aside}: {ex}")
        return {"version": SAVED_VIEWS_VERSION, "folders": {}}


def read_saved_views(path: Path, folder: str) -> SavedViewsResponse:
    """The views saved for `folder`."""
    data = _read_file(path)
    key = folder_key(folder)
    legacy = data.get("legacy_views_folder")
    return SavedViewsResponse(
        views=data["folders"].get(key, []),
        folder=key,
        path=path.as_posix(),
        legacy_views_folder=folder_key(legacy) if legacy else None,
    )


def write_saved_views(
    path: Path, folder: str, views: list[dict[str, Any]]
) -> SavedViewsResponse:
    """Replace the views saved for `folder` (other folders are kept).

    Not thread-safe: the server calls this on its event loop, which
    serializes the read-modify-write.

    Raises:
        ValueError: A view has no name, two views share a name, or the
            file would exceed `MAX_SAVED_VIEWS_BYTES`.
    """
    names = [view.get("name") for view in views]
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("Every saved view needs a non-empty 'name'.")
    if len(set(names)) != len(names):
        raise ValueError("Saved view names must be unique.")

    data = _read_file(path)
    key = folder_key(folder)
    if views:
        data["folders"][key] = views
    else:
        data["folders"].pop(key, None)
    payload = json.dumps(data, indent=2, allow_nan=True)
    if len(payload.encode("utf-8")) > MAX_SAVED_VIEWS_BYTES:
        raise ValueError(
            f"Saved views exceed the {MAX_SAVED_VIEWS_BYTES // (1024 * 1024)} MB limit."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(payload, encoding="utf-8")
    os.replace(temp, path)
    return read_saved_views(path, folder)
