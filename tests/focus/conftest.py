from __future__ import annotations

import os
import shutil
import tempfile

import pytest


@pytest.fixture
def sock_path():
    # macOS caps AF_UNIX paths at 104 bytes and pytest's tmp_path is often longer.
    d = tempfile.mkdtemp(prefix='iaf', dir='/tmp')
    try:
        yield os.path.join(d, 'focus.sock')
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def closers():
    items = []
    yield items
    for item in reversed(items):
        try:
            item.close()
        except Exception:
            pass
