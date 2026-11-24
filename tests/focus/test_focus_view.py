"""`/focus` chooses the page: the live Focus page, or today's page with the reason.

App tier. The real app and the real `FocusView`; `TemplateView.__init__` (database: camera row, latest image) and
`TemplateView.get_context` (status text, astrometry) are replaced by stand-ins. The link side is a real `LinkServer`
when the test wants it up. The admin check is the real `verify_admin_network()`.
"""

from __future__ import annotations

import types

import pytest

from indi_allsky.focus.link import LinkServer
from indi_allsky.focus.rigs import RigDecision
from indi_allsky.focus.rigs import decide


pytestmark = pytest.mark.app

ADMIN_IP = '127.0.0.1'
OUTSIDE_IP = '203.0.113.9'


class _Camera(object):
    id = 1
    name = 'ZWO CCD ASI294MC'


@pytest.fixture
def page(flask_app, sock_path, closers, monkeypatch):
    from indi_allsky.flask import base_views

    state = {'config': {'CAMERA_INTERFACE': 'indi', 'FOCUSER': {}}}

    def init(self, template_name, **kwargs):
        base_views.BaseView.__init__(self, **kwargs)
        self.template_name = template_name
        self.indi_allsky_config = state['config']
        self.camera = _Camera()

    monkeypatch.setattr(base_views.TemplateView, '__init__', init)

    def base_context(self):
        return {
            'website_title': 'test', 'page_title': self.page_title, 'camera_id': self.camera.id,
            'form_camera_select': types.SimpleNamespace(CAMERA_SELECT=lambda **kw: ''),  # the real form queries the database
        }

    monkeypatch.setattr(base_views.TemplateView, 'get_context', base_context)

    class Page(object):
        config = state['config']

        def link_up(self):
            s = LinkServer(sock_path)
            s.start()
            closers.append(s)

        def get(self, remote_addr=ADMIN_IP, admin=True):
            flask_app.set_user(is_admin=admin)
            with flask_app.app.test_client() as client:
                resp = client.get('/indi-allsky/focus', environ_base={'REMOTE_ADDR': remote_addr})
            return resp.status_code, resp.get_data(as_text=True)

    return Page()


def test_classic_for_indi_passive_with_reason(page):
    page.link_up()
    page.config['CAMERA_INTERFACE'] = 'indi_passive'
    reason = decide('indi_passive', _Camera.name, None, True).reason
    assert reason

    status, html = page.get()

    assert status == 200
    assert reason in html
    assert 'id="focus-chart"' in html  # today's page
    assert 'data-focus-session' not in html


def test_classic_when_link_down_with_reason(page):
    reason = decide('indi', _Camera.name, None, False).reason

    status, html = page.get()

    assert status == 200
    assert reason in html
    assert 'id="focus-chart"' in html
    assert 'data-focus-session' not in html


def test_reason_is_html_escaped(page, monkeypatch):
    from indi_allsky.flask import views
    monkeypatch.setattr(views, 'decide', lambda *a, **k: RigDecision('classic', None, '<b>x</b> & y'))

    status, html = page.get()

    assert '&lt;b&gt;x&lt;/b&gt; &amp; y' in html
    assert '<b>x</b>' not in html


def test_classic_page_without_reason_has_no_banner(page, monkeypatch):
    from indi_allsky.flask import views
    monkeypatch.setattr(views, 'decide', lambda *a, **k: RigDecision('classic', None, ''))

    status, html = page.get()

    assert status == 200
    assert 'focus-reason' not in html


def test_new_page_idle_shows_start_only_for_admin_network(page):
    page.link_up()

    status, html = page.get(remote_addr=ADMIN_IP, admin=True)
    assert status == 200
    assert 'data-focus-session' in html
    assert 'data-can-start="1"' in html

    status, html = page.get(remote_addr=OUTSIDE_IP, admin=True)
    assert 'data-focus-session' in html
    assert 'data-can-start="0"' in html

    status, html = page.get(remote_addr=ADMIN_IP, admin=False)
    assert 'data-can-start="0"' in html


def test_new_page_passes_motor_and_ws_url(page):
    page.link_up()
    page.config['FOCUSER'] = {'CLASSNAME': 'focuser_x'}

    status, html = page.get()
    assert 'data-motor="1"' in html
    assert 'data-ws-url="ws://localhost/indi-allsky/ws/focus"' in html

    page.config['FOCUSER'] = {}
    status, html = page.get()
    assert 'data-motor="0"' in html


def test_idle_page_keeps_todays_picture_and_score(page):
    page.link_up()

    status, html = page.get()

    assert 'data-focus-session' in html
    assert 'data-poll-url="/indi-allsky/js/focus"' in html  # focus_session.js polls it while idle
    assert 'id="image-canvas"' in html
    assert 'id="focus-score"' in html
    assert 'id="REFRESH_SELECT"' in html
    assert 'id="ZOOM_SELECT"' in html


def test_new_page_has_the_live_view_and_its_script(page):
    page.link_up()
    page.config['FOCUSER'] = {'CLASSNAME': 'focuser_x'}

    status, html = page.get()

    assert status == 200
    for element in ('fs-idle', 'fs-live', 'fs-canvas', 'fs-rings', 'fs-age', 'fs-stxt', 'fs-value', 'fs-sub',
                    'fs-pick', 'fs-stars-pop', 'fs-snr', 'fs-exp', 'fs-gain', 'fs-apply', 'fs-region-btn',
                    'fs-region', 'fs-warn', 'fs-keep-going', 'fs-done', 'focus-start', 'FOCUS_CCW', 'FOCUS_CW',
                    'STEP_DEGREES', 'fs-toast', 'fs-ended'):
        assert 'id="{0:s}"'.format(element) in html, element
    assert 'data-motor-url="/indi-allsky/ajax/focuscontroller"' in html  # the same POST as today's page
    assert 'js/focus_session.js' in html
    assert 'id="fs-trace"' not in html  # the trace is for hand-turned rigs
    assert 'Start focusing' in html
    assert 'Normal pictures pause while you focus.' in html


def test_new_page_without_motor_has_the_trace(page):
    page.link_up()

    status, html = page.get()

    assert 'id="fs-trace"' in html
    assert 'id="FOCUS_CCW"' not in html


def test_watchers_get_no_controls(page):
    page.link_up()
    page.config['FOCUSER'] = {'CLASSNAME': 'focuser_x'}

    for remote_addr, admin in ((OUTSIDE_IP, True), (ADMIN_IP, False)):
        status, html = page.get(remote_addr=remote_addr, admin=admin)

        assert status == 200
        assert 'No focus session is running. An admin can start one.' in html
        for element in ('focus-start', 'fs-done', 'fs-pick', 'fs-keep-going', 'fs-apply', 'fs-region',
                        'fs-exp-btn', 'fs-region-btn', 'fs-snr', 'fs-stars-chip', 'FOCUS_CCW', 'FOCUS_CW',
                        'STEP_DEGREES'):
            assert 'id="{0:s}"'.format(element) not in html, element
        assert 'js/focus_session.js' in html  # they still watch


def test_stills_rig_shows_the_reason(page):
    page.link_up()
    page.config['CAMERA_INTERFACE'] = 'mqtt_imx477'

    status, html = page.get()

    assert status == 200
    assert 'data-focus-session' in html
    assert 'live video is not available for this camera type' in html
    assert 'id="fs-region-btn"' in html  # the crop moves on stills rigs too (spec 6.2)
