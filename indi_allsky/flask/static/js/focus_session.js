/* The live Focus page (spec section 8): the idle view, and a focus session over /ws/focus.
 *
 * Sections: pure helpers (no DOM; tests/focus/test_focus_page_js.py runs them with node), then, in the browser only:
 * page state, connection and clock, frames, status line, the number, rings and stars, camera and region controls,
 * motor and hand-turned rigs, idle warning, the idle view, and start-up.
 *
 * Message data is never put into the page as HTML: text goes in with textContent, rings are built with
 * createElementNS.
 */
(function () {
    'use strict';

    // ==== pure helpers ===========================================================================================

    var HEADER_SIZE = 29;          // struct '>4sBIdffI' (indi_allsky/focus/frame.py HEADER)
    var HEADER_VERSION = 1;
    var CMD_ID_MAX = 4294967295;   // 2**32 - 1
    var STALL_INTERVALS = 3;       // Stalled after this many expected intervals with no picture
    var STALL_MIN_S = 1.0;         // ... and never sooner than this, so a network hiccup does not flash red
    var RING_PICK_PX = 12;         // a tap picks the nearest ring within this many screen px

    var ENDED_TEXT = {
        'done': 'The focus session ended: Done was pressed.',
        'no_page': 'The focus session ended: no focus page was open for 2 minutes.',
        'idle': 'The focus session ended: nobody touched the page for 15 minutes.',
        'failed': 'The focus session ended because something failed. The indi-allsky log says what.',
        'stopped': 'The focus session ended: the capture program stopped or reloaded.',
        'slow': 'This page was closed because it could not keep up with the pictures. It reconnects by itself.'
    };

    // The 29-byte frame header, big-endian, no padding. Null when it is not a frame this page knows.
    function parseHeader(buf) {
        if (!buf || buf.byteLength < HEADER_SIZE) {
            return null;
        }
        var v = new DataView(buf, 0, HEADER_SIZE);
        if (v.getUint8(0) !== 0x49 || v.getUint8(1) !== 0x41 || v.getUint8(2) !== 0x46 || v.getUint8(3) !== 0x46) {
            return null;  // 'IAFF'
        }
        if (v.getUint8(4) !== HEADER_VERSION) {
            return null;
        }
        return {
            frame_no: v.getUint32(5, false),
            capture_time: v.getFloat64(9, false),
            exposure: v.getFloat32(17, false),
            gain: v.getFloat32(21, false),
            cmd_id: v.getUint32(25, false)
        };
    }

    // Server clock minus page clock (s), from the ping with the shortest round trip: [{rtt, offset}].
    function bestOffset(samples) {
        var best = null;
        for (var i = 0; i < samples.length; i++) {
            if (best === null || samples[i].rtt < best.rtt) {
                best = samples[i];
            }
        }
        return best === null ? null : best.offset;
    }

    // One ping sample: sent and received in page seconds, serverT in server seconds.
    function clockSample(sent, received, serverT) {
        return {rtt: received - sent, offset: serverT - (sent + received) / 2};
    }

    // 'live' | 'waiting' | 'stalled'. sinceFrame: s since the last picture arrived (null: none yet);
    // interval: the expected s between pictures; pending: a command of this page is not in the picture yet.
    function statusState(sinceFrame, interval, pending) {
        if (sinceFrame === null) {
            return 'waiting';
        }
        if (sinceFrame > Math.max(STALL_INTERVALS * interval, STALL_MIN_S)) {
            return 'stalled';
        }
        return pending ? 'waiting' : 'live';
    }

    // Why a session ended: a code from capture, or a full sentence from the web program (shown as it is).
    function endedText(reason) {
        if (reason === null || reason === undefined || reason === '') {
            return '';
        }
        return Object.prototype.hasOwnProperty.call(ENDED_TEXT, reason) ? ENDED_TEXT[reason] : String(reason);
    }

    // The id of the ring nearest (x, y) (picture px) within maxPx screen px; scale is screen px per picture px.
    function nearestRing(rings, x, y, scale, maxPx) {
        var best = null;
        var bestD = Infinity;
        for (var i = 0; i < rings.length; i++) {
            var r = rings[i];
            if (typeof r.x !== 'number' || typeof r.y !== 'number') {
                continue;
            }
            var d = Math.sqrt((r.x - x) * (r.x - x) + (r.y - y) * (r.y - y)) * scale;
            if (d <= maxPx && d < bestD) {
                best = r.id;
                bestD = d;
            }
        }
        return best;
    }

    // Command ids must keep rising across reloads and pages: capture marks each picture with the highest id that
    // had taken effect, so an id below one already seen would clear "not in the picture yet" at once.
    function nextCmdId(lastSent, highestSeen) {
        var n = Math.max(lastSent || 0, highestSeen || 0) + 1;
        return n > CMD_ID_MAX ? 1 : n;
    }

    function fmtAge(s) {
        if (s < 0) {
            s = 0;
        }
        return (s < 1 ? s.toFixed(2) : s < 10 ? s.toFixed(1) : Math.round(s).toString()) + ' s old';
    }

    function fmtExposure(s) {
        if (s >= 1) {
            return (Math.round(s * 10) / 10) + ' s';
        }
        var ms = s * 1000;
        return (ms >= 1 ? Math.round(ms) : Number(ms.toPrecision(2))) + ' ms';
    }

    function fmtGain(g) {
        if (Math.abs(g - Math.round(g)) < 1e-6) {
            return String(Math.round(g));
        }
        return Math.abs(g) < 10 ? String(Number(g.toFixed(2))) : g.toFixed(1);
    }

    // The status line's words when the picture is live (section 6.4). interval: the measured s between pictures
    // (null before two have arrived; the exposure stands in then).
    function liveText(picture, exposure, interval) {
        var valid = typeof exposure === 'number' && exposure > 0;
        if (picture === 'stills') {
            return valid ? 'Still pictures · ' + fmtExposure(exposure) + ' each' : 'Still pictures';
        }
        var every = typeof interval === 'number' && interval > 0 ? Math.max(interval, valid ? exposure : 0) :
            (valid ? exposure : null);
        if (every === null) {
            return 'Live video';
        }
        var rate = 1 / every;
        var n = rate >= 1 ? Math.round(rate) : Number(rate.toFixed(1));
        return 'Live video · ' + n + (n === 1 ? ' picture a second' : ' pictures a second');
    }

    // -- commands not in the picture yet: [{id, kind}], kind 'motor' | 'camera' | 'region'. Each returns a new list.

    function pendingAdd(list, id, kind) {
        return list.concat([{id: id, kind: kind}]);
    }

    // A picture marked cmdId shows every command up to and including it.
    function pendingShown(list, cmdId) {
        return list.filter(function (p) { return p.id > cmdId; });
    }

    function pendingDrop(list, id) {
        return list.filter(function (p) { return p.id !== id; });
    }

    // An error refusing a command: drop that command by its id; with no id, the newest of the command's kind.
    var COMMAND_KIND = {set_camera: 'camera', set_region: 'region', moved: 'motor'};
    function pendingRefused(list, command, cmdId) {
        if (typeof cmdId === 'number') {
            return pendingDrop(list, cmdId);
        }
        var kind = COMMAND_KIND[command];
        for (var i = list.length - 1; kind && i >= 0; i--) {
            if (list[i].kind === kind) {
                return list.slice(0, i).concat(list.slice(i + 1));
            }
        }
        return list;
    }

    // -- the camera controls ----------------------------------------------------------------------------------------

    function known(v) {
        return typeof v === 'number' && isFinite(v);
    }

    // The exposure choices: the ladder inside the camera's limits, plus the current exposure.
    function exposureChoices(ladder, current, min, max) {
        var out = ladder.filter(function (x) {
            return (!known(min) || x >= min) && (!known(max) || x <= max) && Math.abs(x - current) > 1e-9;
        });
        if (known(min) && (!out.length || out[0] > min) && Math.abs(min - current) > 1e-9) {
            out.push(min);
        }
        if (known(max) && Math.abs(max - current) > 1e-9 && out.indexOf(max) < 0) {
            out.push(max);
        }
        out.push(current);
        return out.sort(function (a, b) { return a - b; });
    }

    // The gain slider: the camera's range when known (else 0 .. 600, or twice the current gain), with a fine step
    // when the range is small.
    function gainRange(min, max, current) {
        if (known(min) && known(max) && min === max) {
            // a fixed-gain camera: nothing to choose (capture sends null limits for a camera with no gain control,
            // and the slider then falls back to the defaults below)
            return {min: min, max: max, step: 1, fixed: true};
        }
        // a camera limit is a hard edge (set_camera refuses beyond it); only an unknown end stretches to take in
        // the current gain
        var lo = known(min) ? min : Math.min(0, current);
        var hi = known(max) && max > lo ? max : Math.max(600, Math.ceil(current * 2), lo + 1, current);
        var span = hi - lo;
        return {min: lo, max: hi, step: span < 50 ? span / 100 : 1};
    }

    // The gain to put on the slider and send: inside the span, so a configured gain the camera cannot take is never
    // sent back untouched; a fixed gain is the camera's one value.
    function clampGain(span, g) {
        if (span.fixed) {
            return span.min;
        }
        return Math.min(span.max, Math.max(span.min, g));
    }

    // How long to expect between pictures. Before two have arrived there is no measured gap, and a still picture
    // also takes its download: allow for that so the first picture is not called Stalled.
    var DOWNLOAD_ALLOWANCE_S = 2.0;
    function expectedGap(exposure, gaps) {
        var exp = known(exposure) && exposure > 0 ? exposure : 0.1;
        if (!gaps.length) {
            return exp + DOWNLOAD_ALLOWANCE_S;
        }
        var sorted = gaps.slice().sort(function (a, b) { return a - b; });
        return Math.max(exp, sorted[Math.floor(sorted.length / 2)]);
    }

    // The box on the whole-sensor picture: the current region, or (whole-frame pictures) capture's first-region
    // size centred: capture_session.default_region, a square of at most VIDEO_REGION_PX on the Bayer grid.
    var REGION_PX = 1024;
    function regionBox(W, H, roi) {
        if (Array.isArray(roi) && roi.length === 4) {
            var w = Math.min(W, roi[2]);
            var h = Math.min(H, roi[3]);
            return {x: Math.min(roi[0], W - w), y: Math.min(roi[1], H - h), w: w, h: h};
        }
        var size = Math.min(REGION_PX, W, H);
        size -= size % 2;
        var x = Math.floor((W - size) / 2);
        var y = Math.floor((H - size) / 2);
        return {x: x - x % 2, y: y - y % 2, w: size, h: size};
    }

    // Move region is offered in both modes (spec 6.2): a stills rig starts with no region (whole frame) and can
    // still choose one.
    function canMoveRegion(mode) {
        return !!mode && (mode.picture === 'video' || mode.picture === 'stills');
    }

    var STAR_WORDS = {
        kept: 'measured', weak: 'too faint', dropped: 'left out', pinned: 'the only one measured',
        saturated: 'too bright'
    };
    function starWord(state) {
        return Object.prototype.hasOwnProperty.call(STAR_WORDS, state) ? STAR_WORDS[state] : String(state);
    }

    var pure = {
        HEADER_SIZE: HEADER_SIZE, parseHeader: parseHeader, bestOffset: bestOffset, clockSample: clockSample,
        statusState: statusState, endedText: endedText, nearestRing: nearestRing, nextCmdId: nextCmdId,
        fmtAge: fmtAge, fmtExposure: fmtExposure, fmtGain: fmtGain, liveText: liveText,
        pendingAdd: pendingAdd, pendingShown: pendingShown, pendingDrop: pendingDrop, pendingRefused: pendingRefused,
        exposureChoices: exposureChoices, gainRange: gainRange, clampGain: clampGain, expectedGap: expectedGap, regionBox: regionBox,
        starWord: starWord, canMoveRegion: canMoveRegion
    };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = pure;
    }
    if (typeof document === 'undefined') {
        return;
    }

    // ==== page state =============================================================================================

    var EXPOSURES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75,
                     1, 2, 3, 5, 10, 15, 20, 30, 60];
    var PING_EVERY_MS = 10000;
    var PINGS_PER_BURST = 8;
    var KEEP_GOING_EVERY_MS = 30000;
    var TRACE_S = 60;

    var MSG_WAITING = {
        'motor': 'Lens moving · not in the picture yet',
        'camera': 'New exposure and gain · not in the picture yet',
        'region': 'Region moving · not in the picture yet'
    };
    var MSG_FIRST = 'Waiting for the first picture';
    var MSG_STALLED = 'No new picture · check the camera';
    var MSG_NOT_CONNECTED = 'Not connected to the capture program yet. Try again in a moment.';
    var MSG_MOVE_FAILED = 'The lens did not move.';
    var MSG_WARN = 'Nobody has touched this page for a while. Normal pictures resume in {0} s.';
    var MSG_WARN_WATCHER = 'Nobody is using the focus controls. The focus session ends in {0} s unless an admin ' +
        'keeps it going.';
    var ST = {kept: '#4ade80', weak: '#9ca3af', saturated: '#fbbf24', dropped: '#f87171', pinned: '#60a5fa'};

    var root = document.getElementById('focus-session');
    var $id = function (id) { return document.getElementById(id); };
    var phone = window.matchMedia('(max-width: 767px)');

    var S = {
        ws: null, retry: 0, retryTimer: null, rejoined: false,
        session: 'idle', controls: false, rig: null,
        mode: null, reading: null, stars: [], minSnr: 8, snrHeld: null, snrTimer: null, picking: false,
        drawnFrame: -1, captureTime: null, lastArrival: null, intervals: [], highestCmd: 0,
        // ids start from the clock in seconds, so a new page's first command is above every earlier page's ids
        // (capture marks pictures with the highest id that took effect, from any page)
        lastCmd: Math.floor(Date.now() / 1000),
        offset: 0, haveOffset: false, pingSeq: 0, pingsOut: {}, burst: [], pingTimer: null,
        pending: [],                // [{id, kind}] this page's commands not in the picture yet
        decoding: false, nextFrame: null,
        preview: null,              // {roiKey} while a whole-sensor picture is wanted
        big: false,
        trace: [], best: null, traceMeasure: null,
        warnUntil: null, lastKeep: 0
    };

    function nowS() {
        return Date.now() / 1000;
    }

    function setText(el, text) {
        if (el && el.textContent !== text) {
            el.textContent = text;
        }
    }

    function show(el, on) {
        if (el) {
            el.hidden = !on;
        }
    }

    var toastTimer = null;
    function toast(text) {
        var el = $id('fs-toast');
        setText(el, text);
        show(el, true);
        clearTimeout(toastTimer);
        toastTimer = setTimeout(function () { show(el, false); }, 6000);
    }

    // ==== connection and clock ===================================================================================

    function send(msg) {
        if (!S.ws || S.ws.readyState !== WebSocket.OPEN) {
            return false;
        }
        S.ws.send(JSON.stringify(msg));
        return true;
    }

    function connect() {
        S.retryTimer = null;
        var ws;
        try {
            ws = new WebSocket(root.getAttribute('data-ws-url'));
        } catch (e) {
            reconnectLater();
            return;
        }
        ws.binaryType = 'arraybuffer';
        S.ws = ws;
        ws.onopen = function () {
            S.retry = 0;
            S.rejoined = true;  // the next active state starts the live view afresh: it may be another session
            send({kind: 'hello'});
            pingBurst();
            setStartEnabled();
        };
        ws.onmessage = function (e) {
            if (typeof e.data === 'string') {
                var msg;
                try {
                    msg = JSON.parse(e.data);
                } catch (err) {
                    return;
                }
                if (msg && typeof msg === 'object') {
                    onJson(msg);
                }
            } else {
                onFrame(e.data);
            }
        };
        ws.onclose = function () {
            if (S.ws === ws) {
                S.ws = null;
                // what this page was waiting on may never be answered now (a `moved` that could not be sent, a
                // preview, a pick's progress); the rejoined page starts clean
                S.pending = [];
                S.picking = false;
                closeRegion();
                renderStarsUi();
                if (S.session === 'active') {
                    renderNumber();
                }
                setStartEnabled();
                reconnectLater();
            }
        };
    }

    function reconnectLater() {
        if (S.retryTimer === null) {
            var wait = Math.min(30000, 2000 * Math.pow(2, S.retry));
            S.retry += 1;
            S.retryTimer = setTimeout(connect, wait);
        }
    }

    // Every 10 s: 8 pings one after another; the shortest round trip of a finished burst sets the clock offset.
    // Until the first burst finishes, its best sample so far stands in.
    function pingBurst() {
        if (S.burst.length) {
            S.offset = bestOffset(S.burst);
            S.haveOffset = true;
        }
        S.burst = [];
        var stale = nowS() - PING_EVERY_MS / 1000;
        Object.keys(S.pingsOut).forEach(function (id) {
            if (S.pingsOut[id] < stale) {
                delete S.pingsOut[id];  // a pong that never came
            }
        });
        var sent = 0;
        clearTimeout(S.pingTimer);
        (function one() {
            if (sent < PINGS_PER_BURST && send({kind: 'ping', id: ++S.pingSeq})) {
                S.pingsOut[S.pingSeq] = nowS();
                sent += 1;
                S.pingTimer = setTimeout(one, 150);
            } else {
                S.pingTimer = setTimeout(pingBurst, PING_EVERY_MS);
            }
        })();
    }

    function onPong(msg) {
        var sent = S.pingsOut[msg.id];
        if (sent === undefined || typeof msg.t !== 'number') {
            return;
        }
        delete S.pingsOut[msg.id];
        S.burst.push(clockSample(sent, nowS(), msg.t));
        if (!S.haveOffset || S.burst.length >= PINGS_PER_BURST) {
            S.offset = bestOffset(S.burst);
            S.haveOffset = S.haveOffset || S.burst.length >= PINGS_PER_BURST;
        }
    }

    function onJson(msg) {
        switch (msg.kind) {
        case 'state': onState(msg); break;
        case 'pong': onPong(msg); break;
        case 'mode': onMode(msg); break;
        case 'reading': onReading(msg); break;
        case 'measure': onMeasure(msg); break;
        case 'stars': onStars(msg); break;
        case 'progress': onProgress(msg); break;
        case 'preview': onPreview(msg); break;
        case 'error': onError(msg); break;
        case 'ended': onEnded(msg); break;
        case 'refused': showEnded(msg.reason); break;
        default: break;
        }
    }

    function onState(msg) {
        S.controls = msg.controls === true;
        if (msg.rig && typeof msg.rig === 'object') {
            S.rig = msg.rig;
        }
        if (msg.session === 'active') {
            if (S.session !== 'active' || S.rejoined) {
                enterLive();
            }
            S.rejoined = false;
            S.warnUntil = typeof msg.idle_warning === 'number' ? Date.now() + msg.idle_warning * 1000 : null;
        } else {
            S.rejoined = false;
            if (S.session === 'active') {
                leaveLive();
            }
            if (msg.idle_reason) {
                showEnded(msg.idle_reason);
            }
        }
        renderIdleReason();
        renderControls();
        renderWarn();
    }

    function onError(msg) {
        var text = typeof msg.message === 'string' ? msg.message : '';
        if (text) {
            toast(text);
        }
        // a refused command never takes effect: its sign must not wait for it
        var before = S.pending.length;
        S.pending = pendingRefused(S.pending, msg.command, msg.cmd_id);
        if (S.pending.length !== before && S.session === 'active') {
            renderNumber();
        }
        if (msg.command === 'preview') {
            closeRegion();
        }
        if (msg.command === 'pick_stars') {
            S.picking = false;
            renderStarsUi();
        }
    }

    function onEnded(msg) {
        if (msg.reason === 'slow') {
            toast(endedText('slow'));  // only this page was closed; the session goes on and the page rejoins
            return;
        }
        S.pending = [];
        closeRegion();
        if (S.session === 'active') {
            leaveLive();
        }
        showEnded(msg.reason);
    }

    function showEnded(reason) {
        var el = $id('fs-ended');
        setText(el, endedText(reason));
        show(el, !!el.textContent);
    }

    // ==== frames =================================================================================================

    function onFrame(buf) {
        var h = parseHeader(buf);
        if (!h || S.session !== 'active') {
            return;
        }
        S.highestCmd = Math.max(S.highestCmd, h.cmd_id);
        var t = nowS();
        if (S.lastArrival !== null) {
            S.intervals.push(t - S.lastArrival);
            if (S.intervals.length > 8) {
                S.intervals.shift();
            }
        }
        S.lastArrival = t;
        if (S.decoding) {
            S.nextFrame = [h, buf];  // newest only
            return;
        }
        decode(h, buf);
    }

    function decode(h, buf) {
        S.decoding = true;
        var blob = new Blob([new Uint8Array(buf, HEADER_SIZE)], {type: 'image/jpeg'});
        createImageBitmap(blob).then(function (bmp) {
            if (h.frame_no > S.drawnFrame) {  // a new session (counting from 1 again) resets drawnFrame in enterLive
                var c = $id('fs-canvas');
                if (c.width !== bmp.width || c.height !== bmp.height) {
                    c.width = bmp.width;
                    c.height = bmp.height;
                }
                c.getContext('2d').drawImage(bmp, 0, 0);
                S.drawnFrame = h.frame_no;
                S.captureTime = h.capture_time;
                pictureCmd(h.cmd_id);
            }
            if (bmp.close) {
                bmp.close();
            }
        }).catch(function () {
            // a picture that does not decode is skipped; the next one replaces it
        }).then(function () {
            S.decoding = false;
            var next = S.nextFrame;
            S.nextFrame = null;
            if (next) {
                decode(next[0], next[1]);
            }
        });
    }

    // The picture on screen carries the highest command that had taken effect when it was exposed.
    function pictureCmd(cmdId) {
        var before = S.pending.length;
        S.pending = pendingShown(S.pending, cmdId);
        if (S.pending.length !== before) {
            renderNumber();
        }
    }

    // A command id for msg, and the sign that waits for it. Motor moves are not sent here (the POST moves the lens,
    // `moved` follows it), so they are only marked when the socket is up to send `moved` afterwards.
    function command(msg, kind) {
        var id = nextCmdId(S.lastCmd, S.highestCmd);
        S.lastCmd = id;
        msg.cmd_id = id;
        var connected = kind === 'motor' ? !!S.ws && S.ws.readyState === WebSocket.OPEN : send(msg);
        if (!connected) {
            if (kind !== 'motor') {
                toast(MSG_NOT_CONNECTED);
            }
            return null;
        }
        S.pending = pendingAdd(S.pending, id, kind);
        renderNumber();
        return id;
    }

    // ==== status line ============================================================================================

    function medianInterval() {
        if (!S.intervals.length) {
            return null;
        }
        var sorted = S.intervals.slice().sort(function (a, b) { return a - b; });
        return sorted[Math.floor(sorted.length / 2)];
    }

    function expectedInterval() {
        return expectedGap(S.mode ? S.mode.exposure : null, S.intervals);
    }

    function renderStatus() {
        var since = S.lastArrival === null ? null : nowS() - S.lastArrival;
        var pending = S.pending.length > 0;
        var state = statusState(since, expectedInterval(), pending);
        var text;
        if (state === 'stalled') {
            text = MSG_STALLED;
        } else if (since === null) {
            text = MSG_FIRST;
        } else if (pending) {
            text = MSG_WAITING[S.pending[S.pending.length - 1].kind];
        } else {
            text = S.mode ? liveText(S.mode.picture, S.mode.exposure, medianInterval()) : '';
        }
        var live = $id('fs-live');
        if (live.getAttribute('data-status') !== state) {
            live.setAttribute('data-status', state);
        }
        setText($id('fs-age'), S.captureTime === null ? '' : fmtAge(nowS() + S.offset - S.captureTime));
        setText($id('fs-stxt'), text);
    }

    function renderMode() {
        var m = S.mode;
        var reason = m && m.reason ? String(m.reason) : '';
        var text = '';
        if (m && m.picture === 'stills') {
            text = 'Still pictures: ' + reason;
        } else if (reason && reason.indexOf('Live video') !== 0) {
            text = reason;
        }
        setText($id('fs-mode'), text);
        if (m) {
            setText($id('fs-exp-btn'), 'Exposure ' + fmtExposure(m.exposure) + ' ▾');
            setText($id('fs-gain-btn'), 'Gain ' + fmtGain(m.gain) + ' ▾');
        }
        var regionBtn = $id('fs-region-btn');
        if (regionBtn) {
            show(regionBtn, canMoveRegion(m));
        }
    }

    // ==== the number =============================================================================================

    function onReading(msg) {
        if (typeof msg.cmd_id === 'number') {
            S.highestCmd = Math.max(S.highestCmd, msg.cmd_id);
        }
        S.reading = msg;
        traceAdd(msg);
        renderNumber();
        drawRings();
    }

    function onMeasure(msg) {
        var was = S.traceMeasure;
        if (was && msg.measure !== was && (msg.measure === 'stars' || msg.measure === 'contrast')) {
            var trace = !!$id('fs-trace');  // hand-turned rigs only; motor rigs have no trace in PR 1
            var stars = msg.measure === 'stars';
            setText($id('fs-progress'), trace ?
                (stars ? 'Stars found. The trace starts again with star size.' :
                    'No stars found. The trace starts again with the contrast score.') :
                (stars ? 'Stars found. The number is now star size.' :
                    'No stars found. The number is now the contrast score.'));
        }
        traceReset(msg.measure);
    }

    function renderNumber() {
        var r = S.reading;
        var label = '';
        var value = '–';
        var unit = '';
        var sub = '';
        if (r && r.measure === 'stars' && typeof r.size === 'number') {
            value = r.size.toFixed(2);
            unit = ' px';
            var pinned = (r.rings || []).filter(function (g) { return g.state === 'pinned'; })[0];
            sub = pinned ? 'star ' + (pinned.id + 1) + ' only' :
                'average of ' + r.stars_used + (r.stars_used === 1 ? ' star' : ' stars');
        } else if (r && r.measure === 'contrast' && typeof r.contrast === 'number') {
            label = 'Contrast score ';
            value = r.contrast >= 10 ? String(Math.round(r.contrast)) : r.contrast.toFixed(1);
            sub = 'No stars found · higher is sharper';
        } else if (r && r.message) {
            sub = String(r.message);
        }
        var hasValue = value !== '–';
        var moving = S.pending.length > 0 && hasValue;
        var before = '';
        if (moving) {
            before = S.pending[S.pending.length - 1].kind === 'motor' ? 'from before your move' : 'from before your change';
        } else if (hasValue && r.message) {
            before = String(r.message);
        }
        var v = $id('fs-value');
        setText($id('fs-value-n'), label + value);
        setText($id('fs-value-unit'), unit);
        v.classList.toggle('fs-dim', moving || !hasValue);
        setText($id('fs-sub'), sub);
        $id('fs-sub').classList.toggle('fs-dim', moving);
        setText($id('fs-sub2'), before);
    }

    // ==== rings and stars ========================================================================================

    function starById(id) {
        for (var i = 0; i < S.stars.length; i++) {
            if (S.stars[i].id === id) {
                return S.stars[i];
            }
        }
        return null;
    }

    // The state drawn: while the brightness slider is held, kept and weak follow it here, not the server.
    function shownState(id, state) {
        var s = starById(id);
        if (S.snrHeld !== null && s && (state === 'kept' || state === 'weak') && typeof s.snr1 === 'number') {
            return s.snr1 < S.snrHeld ? 'weak' : 'kept';
        }
        return state;
    }

    function ringList() {
        if (!S.stars.length) {
            return [];
        }
        var r = S.reading;
        if (r && r.rings && r.rings.length) {
            return r.rings;
        }
        return S.stars;
    }

    function pictureSize() {
        var r = S.reading;
        if (r && r.width && r.height) {
            return [r.width, r.height];
        }
        var c = $id('fs-canvas');
        return [c.width * 2, c.height * 2];  // the JPEG is half size
    }

    function svgEl(name, attrs) {
        var el = document.createElementNS('http://www.w3.org/2000/svg', name);
        for (var k in attrs) {
            if (Object.prototype.hasOwnProperty.call(attrs, k)) {
                el.setAttribute(k, attrs[k]);
            }
        }
        return el;
    }

    function drawRings() {
        var svg = $id('fs-rings');
        var wh = pictureSize();
        var vb = '0 0 ' + wh[0] + ' ' + wh[1];
        if (svg.getAttribute('viewBox') !== vb) {
            svg.setAttribute('viewBox', vb);
        }
        while (svg.firstChild) {
            svg.removeChild(svg.firstChild);
        }
        var shown = $id('fs-pic').clientWidth || 1;
        var k = wh[0] / shown;  // picture px per screen px
        var rings = ringList();
        for (var i = 0; i < rings.length; i++) {
            var g = rings[i];
            if (typeof g.x !== 'number' || typeof g.y !== 'number') {
                continue;
            }
            var st = shownState(g.id, g.state);
            var col = ST[st] || ST.kept;
            var rad = (S.big ? 12 : 9) * k;
            var circle = svgEl('circle', {cx: g.x, cy: g.y, r: rad, fill: 'none', stroke: col,
                                          'stroke-width': (st === 'pinned' ? 3 : 2) * k});
            if (st === 'dropped') {
                circle.setAttribute('stroke-dasharray', (5 * k) + ' ' + (4 * k));
            } else if (st === 'weak') {
                circle.setAttribute('stroke-dasharray', (2 * k) + ' ' + (4 * k));
            }
            svg.appendChild(circle);
            var label = svgEl('text', {x: g.x + rad + 2 * k, y: g.y - rad + 4 * k, fill: col,
                                       'font-size': 13 * k, 'font-weight': 700, 'paint-order': 'stroke',
                                       stroke: '#000', 'stroke-width': 3 * k});
            label.textContent = String(g.id + 1);
            svg.appendChild(label);
        }
    }

    function onStars(msg) {
        S.stars = Array.isArray(msg.stars) ? msg.stars : [];
        if (typeof msg.min_snr === 'number') {
            S.minSnr = msg.min_snr;
            if (S.snrHeld !== null && Math.abs(S.snrHeld - msg.min_snr) < 1e-6) {
                S.snrHeld = null;
            }
        }
        if (!S.stars.length && S.reading) {
            S.reading.rings = [];
        }
        traceReset(S.traceMeasure);  // a new star set: sizes before it do not compare
        renderStarsUi();
        drawRings();
    }

    function onProgress(msg) {
        setText($id('fs-progress'), typeof msg.message === 'string' ? msg.message : '');
        if (msg.what === 'pick') {
            S.picking = msg.done !== true;
            renderStarsUi();
        }
    }

    // keep -> drop -> pin -> keep (amendment 9: tap to drop, tap again to pin)
    function nextStarState(state) {
        if (state === 'dropped') {
            return 'pin';
        }
        return state === 'pinned' ? 'keep' : 'drop';
    }

    function tapStar(id) {
        var s = starById(id);
        if (s && S.controls) {
            send({kind: 'star', id: id, state: nextStarState(s.state)});
        }
    }

    function renderStarsUi() {
        var c = S.controls && S.session === 'active';
        show($id('fs-pick'), c && !S.stars.length);
        show($id('fs-stars-chip'), c && S.stars.length > 0);
        if ($id('fs-pick')) {
            $id('fs-pick').disabled = S.picking;
            $id('fs-repick').disabled = S.picking;
        }
        if (!c || !S.stars.length) {
            show($id('fs-stars-pop'), false);
        }
        var range = $id('fs-snr');
        if (range && S.snrHeld === null) {
            range.value = S.minSnr;
            setText($id('fs-snr-v'), String(Math.round(S.minSnr)));
        }
        var list = $id('fs-stars-list');
        if (!list) {
            return;
        }
        while (list.firstChild) {
            list.removeChild(list.firstChild);
        }
        S.stars.forEach(function (s) {
            var li = document.createElement('li');
            var b = document.createElement('button');
            b.type = 'button';
            b.className = 'tw:btn tw:btn-xs tw:btn-ghost';
            var st = shownState(s.id, s.state);
            b.style.color = ST[st] || '';
            b.textContent = 'Star ' + (s.id + 1) + ' · S/N ' + (typeof s.snr1 === 'number' ? s.snr1.toFixed(0) : '?') +
                ' · ' + starWord(st);
            b.addEventListener('click', function () { tapStar(s.id); });
            li.appendChild(b);
            list.appendChild(li);
        });
    }

    function onPictureTap(e) {
        if (phone.matches && !S.big) {
            setBig(true);
            return;
        }
        if (!S.controls || !S.stars.length) {
            return;
        }
        var rect = $id('fs-canvas').getBoundingClientRect();
        var wh = pictureSize();
        var scale = rect.width / wh[0];
        var x = (e.clientX - rect.left) / scale;
        var y = (e.clientY - rect.top) / scale;
        var id = nearestRing(ringList(), x, y, scale, RING_PICK_PX);
        if (id !== null) {
            tapStar(id);
        }
    }

    function setBig(on) {
        S.big = on;
        $id('fs-live').classList.toggle('fs-big', on);
        show($id('fs-shrink'), on);
        drawRings();
        drawTrace();
    }

    // ==== camera and region controls =============================================================================

    var expChoices = EXPOSURES.slice();
    var gainSpan = {min: 0, max: 600, step: 1};
    var camTouched = {exp: false, gain: false};

    function openCamera() {
        var pop = $id('fs-cam-pop');
        var opening = pop.hidden;
        show(pop, opening);
        $id('fs-exp-btn').setAttribute('aria-expanded', String(opening));
        $id('fs-gain-btn').setAttribute('aria-expanded', String(opening));
        if (!opening || !S.mode) {
            return;
        }
        // the camera's own value is always on the slider, so Apply without touching it sends it back unchanged
        var m = S.mode;
        var exp = m.exposure;
        var gain = m.gain;
        // the camera's limits come in `mode`; a limit capture does not know is null and the defaults stand
        expChoices = exposureChoices(EXPOSURES, exp, m.exposure_min, m.exposure_max);
        var r = $id('fs-exp');
        r.max = expChoices.length - 1;
        r.value = expChoices.indexOf(exp);
        gainSpan = gainRange(m.gain_min, m.gain_max, gain);
        var g = $id('fs-gain');
        g.min = gainSpan.min;
        g.max = gainSpan.max;
        g.step = gainSpan.step;
        g.value = clampGain(gainSpan, gain);
        g.disabled = !!gainSpan.fixed;  // a fixed-gain camera: the value shows, nothing to slide
        g.title = gainSpan.fixed ? 'This camera has a fixed gain.' : '';
        camTouched = {exp: false, gain: false};
        showCamera();
    }

    function chosenGain() {
        if (!camTouched.gain || gainSpan.fixed) {
            return clampGain(gainSpan, S.mode.gain);
        }
        var v = Number($id('fs-gain').value);
        // whole numbers on a wide range; two decimals on a narrow one (step = range / 100)
        v = gainSpan.step >= 1 ? Math.round(v) : Math.round(v * 100) / 100;
        return clampGain(gainSpan, v);
    }

    function showCamera() {
        setText($id('fs-exp-v'), fmtExposure(expChoices[Number($id('fs-exp').value)]));
        setText($id('fs-gain-v'), fmtGain(chosenGain()));
    }

    function applyCamera() {
        if (!S.mode) {
            return;
        }
        var exp = camTouched.exp ? expChoices[Number($id('fs-exp').value)] : S.mode.exposure;
        command({kind: 'set_camera', exposure: exp, gain: chosenGain()}, 'camera');
        show($id('fs-cam-pop'), false);
    }

    function roiKey(roi) {
        return Array.isArray(roi) ? roi.join(',') : 'none';
    }

    function onMode(msg) {
        if (S.preview && S.mode && roiKey(msg.roi) !== S.preview.roiKey) {
            closeRegion();  // the region moved: capture drops a whole-sensor picture it was taking, with no reply
        }
        S.mode = msg;
        renderMode();
    }

    function openRegion() {
        if (S.preview || !canMoveRegion(S.mode)) {
            return;
        }
        if (!send({kind: 'preview'})) {
            toast(MSG_NOT_CONNECTED);
            return;
        }
        S.preview = {roiKey: roiKey(S.mode.roi)};
        show($id('fs-region'), true);
        show($id('fs-region-wait'), true);
        show($id('fs-region-pic'), false);
        show($id('fs-region-hint'), false);
    }

    function closeRegion() {
        S.preview = null;
        show($id('fs-region'), false);
    }

    var box = null;  // {x, y, w, h, W, H} in sensor px

    function onPreview(msg) {
        if (!S.preview || typeof msg.jpeg !== 'string' || !/^[A-Za-z0-9+/=]+$/.test(msg.jpeg)) {
            return;
        }
        var W = msg.width;
        var H = msg.height;
        if (!known(W) || !known(H) || W < 2 || H < 2) {
            closeRegion();
            return;
        }
        // the current region, or for a whole-frame picture (stills, or a sensor size capture did not know) the
        // size capture would start a region at, centred
        box = regionBox(W, H, S.mode ? S.mode.roi : null);
        box.W = W;
        box.H = H;
        $id('fs-region-img').src = 'data:image/jpeg;base64,' + msg.jpeg;
        placeBox();
        show($id('fs-region-wait'), false);
        show($id('fs-region-pic'), true);
        show($id('fs-region-hint'), true);
    }

    function placeBox() {
        var el = $id('fs-region-box');
        el.style.left = (box.x / box.W * 100) + '%';
        el.style.top = (box.y / box.H * 100) + '%';
        el.style.width = (box.w / box.W * 100) + '%';
        el.style.height = (box.h / box.H * 100) + '%';
    }

    function bindRegionDrag() {
        var el = $id('fs-region-box');
        var from = null;
        el.addEventListener('pointerdown', function (e) {
            if (!box) {
                return;
            }
            from = {cx: e.clientX, cy: e.clientY, x: box.x, y: box.y};
            try { el.setPointerCapture(e.pointerId); } catch (err) { /* not every browser */ }
            e.preventDefault();
        });
        el.addEventListener('pointermove', function (e) {
            if (!from) {
                return;
            }
            var k = box.W / ($id('fs-region-img').clientWidth || 1);
            box.x = Math.max(0, Math.min(box.W - box.w, from.x + (e.clientX - from.cx) * k));
            box.y = Math.max(0, Math.min(box.H - box.h, from.y + (e.clientY - from.cy) * k));
            placeBox();
        });
        el.addEventListener('pointerup', function () {
            if (!from) {
                return;
            }
            from = null;
            var even = function (v) { return Math.max(0, Math.floor(v / 2) * 2); };
            var roi = [even(box.x), even(box.y), even(box.w), even(box.h)];
            closeRegion();
            command({kind: 'set_region', roi: roi}, 'region');
        });
        el.addEventListener('pointercancel', function () { from = null; });
    }

    // ==== motor and hand-turned rigs =============================================================================

    // The same POST today's page (focus.html) makes; the web process moves the lens (section 9.7 A).
    function focuserMove(direction, degrees) {
        var ccw = $id('FOCUS_CCW');
        var cw = $id('FOCUS_CW');
        ccw.disabled = cw.disabled = true;
        var id = command({}, 'motor');
        var data = {
            'DIRECTION': direction,
            'STEP_DEGREES': degrees
        };
        $.ajax({
            type: 'POST',
            url: root.getAttribute('data-motor-url'),
            contentType: 'application/json',
            data: JSON.stringify(data),
            success: function () {
                ccw.disabled = cw.disabled = false;
                // pictures exposed from 0.5 s after this reaches capture carry the id (section 9.7 A); a `moved`
                // that cannot be sent would never show, so its sign goes
                if (id !== null && !send({kind: 'moved', cmd_id: id})) {
                    S.pending = pendingDrop(S.pending, id);
                    renderNumber();
                }
            },
            error: function (resp) {
                ccw.disabled = cw.disabled = false;
                if (id !== null) {
                    S.pending = pendingDrop(S.pending, id);
                    renderNumber();
                }
                var text = MSG_MOVE_FAILED;
                try {
                    var errors = JSON.parse(resp.responseText);
                    if (errors && errors.focuser_error && errors.focuser_error[0]) {
                        text = String(errors.focuser_error[0]);
                    }
                } catch (err) {
                    // not JSON: the plain message
                }
                toast(text);
            }
        });
    }

    function traceReset(measure) {
        S.trace = [];
        S.best = null;
        S.traceMeasure = measure || null;
        drawTrace();
    }

    function traceAdd(r) {
        var v = null;
        if (r.measure === 'stars' && typeof r.size === 'number') {
            v = r.size;
        } else if (r.measure === 'contrast' && typeof r.contrast === 'number') {
            v = r.contrast;
        }
        if (r.measure !== S.traceMeasure) {
            traceReset(r.measure);
        }
        if (v === null || typeof r.capture_time !== 'number') {
            return;
        }
        S.trace.push([r.capture_time, v]);
        while (S.trace.length && S.trace[0][0] < r.capture_time - TRACE_S) {
            S.trace.shift();
        }
        var higher = r.measure === 'contrast';  // best: the smallest star size, the highest contrast score
        if (S.best === null || (higher ? v > S.best : v < S.best)) {
            S.best = v;
        }
        drawTrace();
    }

    function drawTrace() {
        var c = $id('fs-trace');
        if (!c || !c.clientWidth) {
            return;
        }
        var dpr = window.devicePixelRatio || 1;
        var W = c.clientWidth;
        var H = c.clientHeight;
        if (c.width !== Math.round(W * dpr)) {
            c.width = Math.round(W * dpr);
            c.height = Math.round(H * dpr);
        }
        var ctx = c.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, W, H);
        var pts = S.trace;
        var contrast = S.traceMeasure === 'contrast';
        var best = $id('fs-best');
        setText(best, S.best === null ? '' : 'Best so far: ' + (contrast ?
            'contrast score ' + Math.round(S.best) : S.best.toFixed(2) + ' px'));
        if (!pts.length) {
            return;
        }
        var lo = S.best;
        var hi = S.best;
        pts.forEach(function (p) { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); });
        var pad = Math.max((hi - lo) * 0.15, contrast ? 1 : 0.05);
        lo -= pad;
        hi += pad;
        var t1 = pts[pts.length - 1][0];
        var xOf = function (t) { return 4 + (W - 8) * (1 - (t1 - t) / TRACE_S); };
        var yOf = function (v) { return 6 + (H - 12) * (1 - (v - lo) / (hi - lo)); };
        ctx.strokeStyle = '#4ade80';
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(0, yOf(S.best));
        ctx.lineTo(W, yOf(S.best));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = '#60a5fa';
        ctx.lineWidth = 2;
        ctx.beginPath();
        pts.forEach(function (p, i) {
            if (i) {
                ctx.lineTo(xOf(p[0]), yOf(p[1]));
            } else {
                ctx.moveTo(xOf(p[0]), yOf(p[1]));
            }
        });
        ctx.stroke();
    }

    // ==== idle warning ===========================================================================================

    function renderWarn() {
        var on = S.session === 'active' && S.warnUntil !== null;
        show($id('fs-warn'), on);
        if (on) {
            var left = Math.max(0, Math.ceil((S.warnUntil - Date.now()) / 1000));
            setText($id('fs-warn-text'), (S.controls ? MSG_WARN : MSG_WARN_WATCHER).replace('{0}', String(left)));
        }
    }

    function keepGoing(force) {
        if (S.session !== 'active' || !S.controls) {
            return;
        }
        var now = Date.now();
        if (force || now - S.lastKeep >= KEEP_GOING_EVERY_MS) {
            if (send({kind: 'keep_going'})) {
                S.lastKeep = now;
            }
        }
    }

    // ==== live and idle views ====================================================================================

    function enterLive() {
        S.session = 'active';
        S.drawnFrame = -1;
        S.captureTime = null;
        S.lastArrival = null;
        S.intervals = [];
        S.pending = [];
        S.reading = null;
        S.stars = [];
        S.picking = false;
        S.lastKeep = 0;
        traceReset(null);
        setText($id('fs-progress'), '');
        show($id('fs-ended'), false);
        show($id('fs-idle'), false);
        show($id('fs-live'), true);
        renderNumber();
        renderStarsUi();
        drawRings();
    }

    function leaveLive() {
        S.session = 'idle';
        S.warnUntil = null;
        S.pending = [];
        closeRegion();
        setBig(false);
        show($id('fs-cam-pop'), false);
        show($id('fs-live'), false);
        show($id('fs-idle'), true);
        loadNextImage();
    }

    function renderControls() {
        renderStarsUi();
    }

    function renderIdleReason() {
        var reason = root.getAttribute('data-reason') || '';
        if (S.rig && typeof S.rig.reason === 'string' && S.rig.reason) {
            reason = S.rig.reason;
        }
        if (S.rig && S.rig.picture === 'stills' && reason) {
            reason = 'Still pictures: ' + reason;
        }
        setText($id('fs-idle-reason'), reason);
        show($id('fs-idle-reason'), !!reason);
    }

    function setStartEnabled() {
        var b = $id('focus-start');
        if (b) {
            b.disabled = !S.ws || S.ws.readyState !== WebSocket.OPEN;
        }
    }

    // Today's picture and score while no session runs: the same poll as today's page (focus.html).
    var pollTimer = null;
    function loadNextImage() {
        clearTimeout(pollTimer);
        var refresh = Number($('#REFRESH_SELECT').val()) * 1000 || 5000;
        pollTimer = setTimeout(loadNextImage, refresh);
        if (S.session === 'active') {
            return;
        }
        $.ajax({
            type: 'GET',
            url: root.getAttribute('data-poll-url'),
            contentType: 'application/json',
            data: {'zoom': $('#ZOOM_SELECT').val(), 'x_offset': $('#X_OFFSET').val(), 'y_offset': $('#Y_OFFSET').val()},
            success: function (rdata) {
                if (S.session === 'active' || !rdata) {
                    return;
                }
                setText($id('focus-score'), Number(rdata['blur_score'] || 0).toFixed(2));
                setText($id('focus-stars'), String(Number(rdata['star_count'] || 0)));
                if (rdata['image_b64'] && /^[A-Za-z0-9+/=]+$/.test(rdata['image_b64'])) {
                    var img = new Image();
                    img.onload = function () {
                        var c = $id('image-canvas');
                        c.width = this.width;
                        c.height = this.height;
                        c.getContext('2d').drawImage(this, 0, 0);
                    };
                    img.src = 'data:image/jpeg;base64,' + rdata['image_b64'];
                }
            }
        });
    }

    // ==== start-up ===============================================================================================

    function on(id, event, fn) {
        var el = $id(id);
        if (el) {
            el.addEventListener(event, fn);
        }
    }

    function init() {
        on('focus-start', 'click', function () {
            if (!send({kind: 'start'})) {
                toast(MSG_NOT_CONNECTED);
            }
        });
        on('fs-done', 'click', function () { send({kind: 'done'}); });
        on('fs-keep-going', 'click', function () { keepGoing(true); });
        on('fs-pick', 'click', function () { send({kind: 'pick_stars'}); });
        on('fs-repick', 'click', function () { send({kind: 'pick_stars'}); show($id('fs-stars-pop'), false); });
        on('fs-stars-chip', 'click', function () {
            var pop = $id('fs-stars-pop');
            show(pop, pop.hidden);
            this.setAttribute('aria-expanded', String(!pop.hidden));
        });
        on('fs-snr', 'input', function () {
            S.snrHeld = Number(this.value);
            setText($id('fs-snr-v'), this.value);
            clearTimeout(S.snrTimer);
            drawRings();
            renderStarsUi();
        });
        on('fs-snr', 'change', function () {
            var v = Number(this.value);
            send({kind: 'min_snr', value: v});
            clearTimeout(S.snrTimer);
            S.snrTimer = setTimeout(function () { S.snrHeld = null; renderStarsUi(); drawRings(); }, 2000);
        });
        on('fs-exp-btn', 'click', openCamera);
        on('fs-gain-btn', 'click', openCamera);
        on('fs-cam-close', 'click', function () { show($id('fs-cam-pop'), false); });
        on('fs-exp', 'input', function () { camTouched.exp = true; showCamera(); });
        on('fs-gain', 'input', function () { camTouched.gain = true; showCamera(); });
        on('fs-apply', 'click', applyCamera);
        on('fs-region-btn', 'click', openRegion);
        on('fs-region-cancel', 'click', closeRegion);
        if ($id('fs-region-box')) {
            bindRegionDrag();
        }
        on('fs-pic', 'click', onPictureTap);
        on('fs-shrink', 'click', function (e) { e.stopPropagation(); setBig(false); });
        on('FOCUS_CCW', 'click', function () { focuserMove('ccw', $('#STEP_DEGREES').val()); });
        on('FOCUS_CW', 'click', function () { focuserMove('cw', $('#STEP_DEGREES').val()); });
        ['REFRESH_SELECT', 'ZOOM_SELECT'].forEach(function (id) { on(id, 'change', loadNextImage); });

        // any click or key press on an admin's page counts as touching it, at most once every 30 s
        document.addEventListener('click', function () { keepGoing(false); }, true);
        document.addEventListener('keydown', function () { keepGoing(false); }, true);

        window.addEventListener('resize', function () { drawRings(); drawTrace(); });
        setInterval(function () {
            if (S.session === 'active') {
                renderStatus();
                renderWarn();
            }
        }, 100);

        renderIdleReason();
        setStartEnabled();
        loadNextImage();
        connect();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
