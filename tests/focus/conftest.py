from __future__ import annotations

import importlib
import itertools
import os
import shutil
import sys
import tempfile
import types

import pytest


_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STUBBED = ('PyIndi', 'ephem', 'dateutil', 'dateutil.parser', 'indi_allsky.flask', 'indi_allsky.camera')


def _fake_pyindi():
    counter = itertools.count(1000)
    consts = {}

    def constant(name):
        if name.startswith('__'):
            raise AttributeError(name)
        if name not in consts:
            consts[name] = next(counter)
        return consts[name]

    class _AutoConstants(type):
        def __getattr__(cls, name):
            return constant(name)

    class BaseDevice(object, metaclass=_AutoConstants):
        pass

    class BaseMediator(object):
        pass

    class BaseClient(object):
        def __init__(self, *args, **kwargs):
            pass

    mod = types.ModuleType('PyIndi')
    mod.BaseDevice = BaseDevice
    mod.BaseMediator = BaseMediator
    mod.BaseClient = BaseClient
    mod.__getattr__ = constant
    return mod


@pytest.fixture
def fake_pyindi():
    """Import `indi_allsky.camera.<name>` without PyIndi, Flask or the camera package's other modules.

    Installs a fake `PyIndi` (auto-numbered constants), a fake `indi_allsky.flask` whose `create_app()` returns None,
    stand-ins for `ephem` and `dateutil` when they are not installed, and an `indi_allsky.camera` package that does not
    run its `__init__`. Returns `load(name)`. Everything it put in `sys.modules` is removed afterwards.
    """
    import indi_allsky

    before = set(sys.modules)
    saved = {name: sys.modules[name] for name in _STUBBED if name in sys.modules}
    had_camera_attr = hasattr(indi_allsky, 'camera')
    old_camera_attr = getattr(indi_allsky, 'camera', None)

    sys.modules['PyIndi'] = _fake_pyindi()

    flask = types.ModuleType('indi_allsky.flask')
    flask.create_app = lambda *a, **k: None
    sys.modules['indi_allsky.flask'] = flask

    for name in ('ephem', 'dateutil', 'dateutil.parser'):
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)
    if not hasattr(sys.modules['dateutil'], 'parser'):
        sys.modules['dateutil'].parser = sys.modules['dateutil.parser']

    camera = types.ModuleType('indi_allsky.camera')
    camera.__path__ = [os.path.join(_REPO, 'indi_allsky', 'camera')]
    sys.modules['indi_allsky.camera'] = camera

    def load(name):
        return importlib.import_module('indi_allsky.camera.' + name)

    try:
        yield load
    finally:
        for name in set(sys.modules) - before:
            if name in _STUBBED or name.startswith(('indi_allsky.camera', 'indi_allsky.flask', 'indi_allsky.utils')):
                del sys.modules[name]
        for name in _STUBBED:
            if name in saved:
                sys.modules[name] = saved[name]
            else:
                sys.modules.pop(name, None)
        if had_camera_attr:
            indi_allsky.camera = old_camera_attr
        elif hasattr(indi_allsky, 'camera'):
            del indi_allsky.camera
        for attr in ('flask', 'utils'):
            if 'indi_allsky.' + attr not in sys.modules and hasattr(indi_allsky, attr):
                delattr(indi_allsky, attr)


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
