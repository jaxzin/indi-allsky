"""The link between the capture process and the web process during a focus session.

The capture process runs a ``LinkServer`` on a Unix socket. Each focus page in the web
process holds a ``LinkClient``. Frames and JSON travel server to client; JSON commands
travel client to server.

Wire format, both directions: one kind byte (``b'J'`` JSON, ``b'F'`` frame), a 4-byte
big-endian payload length, then the payload. Clients may only send ``b'J'``.

Message contents are never logged.
"""

from __future__ import annotations

import collections
import errno
import json
import logging
import os
import selectors
import socket
import stat
import struct
import threading
import time

from indi_allsky.focus.session import MAX_PAGES

logger = logging.getLogger('indi_allsky')


KIND_JSON = b'J'
KIND_FRAME = b'F'
_HEADER = struct.Struct('>cI')

CLIENT_JSON_MAX = 64 * 1024        # client -> server command size cap
SERVER_MSG_MAX = 64 * 1024 * 1024  # server -> client frame/JSON size cap

SEND_TIMEOUT_S = 5.0
PROBE_TIMEOUT_S = 0.5
JOIN_TIMEOUT_S = 1.0
_ACCEPT_POLL_S = 0.2
_REFUSAL_TIMEOUT_S = 0.5

REFUSAL = {'kind': 'refused', 'reason': 'Three pages are already watching this focus session.'}


class LinkError(Exception):
    pass


class LinkInUse(LinkError):
    """Another live server already answers on the socket path."""


class LinkUnavailable(LinkError):
    """The client could not reach the server, or the server went away."""

    def __init__(self, reason):
        super(LinkUnavailable, self).__init__(reason)
        self.reason = reason


class SlotOverflow(LinkError):
    """A client's JSON queue reached its cap; the client is too slow to keep."""


def _header(kind, payload):
    return _HEADER.pack(kind, len(payload))


def _encode_json(obj):
    return json.dumps(obj, separators=(',', ':')).encode('utf-8')


class NewestSlot(object):
    """A per-client outbox: JSON is queued in order, frames are newest-only.

    A new frame replaces any unsent frame and goes to the tail, so JSON queued before it
    is still delivered before it.

    The slot does not inspect what it holds: ``LinkServer`` queues already-encoded JSON
    bytes so a broadcast is encoded once.
    """

    def __init__(self, json_cap: int = 256):
        self.json_cap = json_cap
        self._cond = threading.Condition()
        self._queue = collections.deque()  # entries: ('json', dict) or ('frame', bytes)
        self._json_count = 0
        self._closed = False

    def put_frame(self, frame: bytes) -> None:
        with self._cond:
            if self._closed:
                return
            for entry in self._queue:
                if entry[0] == 'frame':
                    self._queue.remove(entry)
                    break
            self._queue.append(('frame', frame))
            self._cond.notify()

    def put_json(self, obj: dict) -> None:
        with self._cond:
            if self._closed:
                return
            if self._json_count >= self.json_cap:
                raise SlotOverflow('json queue full')
            self._queue.append(('json', obj))
            self._json_count += 1
            self._cond.notify()

    def get(self, timeout: float | None = None):
        """Next entry, oldest first; None on timeout or once closed."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._queue and not self._closed:
                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(remaining)
            if self._closed:
                return None
            entry = self._queue.popleft()
            if entry[0] == 'json':
                self._json_count -= 1
            return entry

    def _close(self):
        with self._cond:
            self._closed = True
            self._queue.clear()
            self._json_count = 0
            self._cond.notify_all()


class _Conn(object):
    """One accepted client on the server side."""

    def __init__(self, cid, sock):
        self.cid = cid
        self.sock = sock
        self.slot = NewestSlot()
        self.closed = threading.Event()
        self.threads = []
        self._lock = threading.Lock()
        self._users = 0  # threads still using the socket; the last one closes the fd

    def add_user(self):
        with self._lock:
            self._users += 1

    def release_user(self):
        with self._lock:
            self._users -= 1
            last = self._users == 0
        if last:
            try:
                self.sock.close()
            except OSError:
                pass


class LinkServer(object):
    """The capture-side end of the link."""

    def __init__(self, path: str, max_clients: int = MAX_PAGES):
        self.path = path
        self.max_clients = max_clients
        self._lock = threading.Lock()
        self._clients = {}   # cid -> _Conn
        self._commands = []  # (cid, dict)
        self._next_cid = 1
        self._listener = None
        self._identity = None  # (st_dev, st_ino) of the socket file we bound
        self._stop = threading.Event()
        self._accept_thread = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self._clear_path()

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            # umask is process-wide: other threads creating files during this window
            # also get 0o177. The window is just the bind() call.
            old_umask = os.umask(0o177)
            try:
                listener.bind(self.path)
            finally:
                os.umask(old_umask)
            os.chmod(self.path, 0o600)
            st = os.stat(self.path)
            listener.listen(8)
            listener.settimeout(_ACCEPT_POLL_S)
        except Exception:
            listener.close()
            raise

        self._identity = (st.st_dev, st.st_ino)
        self._listener = listener
        self._stop.clear()
        self._accept_thread = threading.Thread(target=self._accept_loop, name='focus-link-accept', daemon=True)
        self._accept_thread.start()
        logger.info('Focus link listening')

    def _clear_path(self):
        try:
            st = os.lstat(self.path)
        except FileNotFoundError:
            return

        if stat.S_ISDIR(st.st_mode):
            raise IsADirectoryError(errno.EISDIR, 'Focus link path is a directory', self.path)

        if stat.S_ISSOCK(st.st_mode):
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(PROBE_TIMEOUT_S)
            try:
                s.connect(self.path)
            except (ConnectionRefusedError, FileNotFoundError):
                pass  # stale: nobody is listening
            else:
                raise LinkInUse(self.path)
            finally:
                s.close()

        logger.info('Focus link removing stale socket file')
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def close(self) -> None:
        self._stop.set()

        if self._accept_thread is not None and self._accept_thread is not threading.current_thread():
            self._accept_thread.join(JOIN_TIMEOUT_S)
        self._accept_thread = None

        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None

        with self._lock:
            conns = list(self._clients.values())
        for conn in conns:
            self._drop(conn, 'server closing')
        me = threading.current_thread()
        for conn in conns:
            for t in conn.threads:
                if t is not me:
                    t.join(JOIN_TIMEOUT_S)

        if self._identity is not None:
            try:
                st = os.stat(self.path)
                if (st.st_dev, st.st_ino) == self._identity:
                    os.unlink(self.path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning('Focus link could not remove its socket file')
            self._identity = None

    # -- public API --------------------------------------------------------------

    def clients(self) -> list:
        with self._lock:
            return sorted(self._clients)

    def poll_commands(self) -> list:
        with self._lock:
            commands, self._commands = self._commands, []
        return commands

    def broadcast_frame(self, frame: bytes) -> None:
        if len(frame) > SERVER_MSG_MAX:
            raise ValueError('frame larger than the link allows')
        with self._lock:
            conns = list(self._clients.values())
        for conn in conns:
            conn.slot.put_frame(frame)

    def broadcast_json(self, obj: dict) -> None:
        payload = _encode_json(obj)
        if len(payload) > SERVER_MSG_MAX:
            raise ValueError('message larger than the link allows')
        with self._lock:
            conns = list(self._clients.values())
        for conn in conns:
            self._put_json(conn, payload)

    def send_json(self, client_id: int, obj: dict) -> None:
        payload = _encode_json(obj)
        if len(payload) > SERVER_MSG_MAX:
            raise ValueError('message larger than the link allows')
        with self._lock:
            conn = self._clients.get(client_id)
        if conn is None:
            logger.debug('Focus link: no client %d to send to', client_id)
            return
        self._put_json(conn, payload)

    # -- internals ---------------------------------------------------------------

    def _put_json(self, conn, payload):
        # The slot holds encoded bytes so a broadcast encodes once.
        try:
            conn.slot.put_json(payload)
        except SlotOverflow:
            self._drop(conn, 'json queue full')

    def _drop(self, conn, reason):
        """Disconnect a client. Idempotent; safe from any thread."""
        with self._lock:
            if conn.closed.is_set():
                return
            conn.closed.set()
            self._clients.pop(conn.cid, None)
        logger.info('Focus link client %d dropped: %s', conn.cid, reason)
        conn.slot._close()
        try:
            conn.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # macOS raises ENOTCONN once the peer is gone

    def _accept_loop(self):
        while not self._stop.is_set():
            listener = self._listener
            if listener is None:
                return
            try:
                sock, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                logger.exception('Focus link accept failed')
                time.sleep(_ACCEPT_POLL_S)
                continue

            with self._lock:
                full = len(self._clients) >= self.max_clients
                if not full:
                    cid = self._next_cid
                    self._next_cid += 1
                    conn = _Conn(cid, sock)
                    self._clients[cid] = conn

            if full:
                self._refuse(sock)
                continue

            sock.settimeout(SEND_TIMEOUT_S)
            reader = threading.Thread(target=self._reader, args=(conn,), name='focus-link-read-%d' % cid, daemon=True)
            sender = threading.Thread(target=self._sender, args=(conn,), name='focus-link-send-%d' % cid, daemon=True)
            conn.threads = [reader, sender]
            conn.add_user()
            conn.add_user()
            logger.info('Focus link client %d connected', cid)
            reader.start()
            sender.start()

    def _refuse(self, sock):
        logger.info('Focus link refused a client: %d pages already watching', self.max_clients)
        payload = _encode_json(REFUSAL)
        try:
            sock.settimeout(_REFUSAL_TIMEOUT_S)
            sock.sendall(_header(KIND_JSON, payload) + payload)
        except OSError:
            pass
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()

    def _recv_exact(self, conn, n):
        """Read n bytes. None if the client closed or is being dropped."""
        buf = bytearray()
        while len(buf) < n:
            if conn.closed.is_set():
                return None
            try:
                chunk = conn.sock.recv(n - len(buf))
            except socket.timeout:
                # The socket's timeout exists for the sender; an idle reader keeps waiting.
                continue
            except OSError:
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _reader(self, conn):
        reason = 'disconnected'
        try:
            while True:
                header = self._recv_exact(conn, _HEADER.size)
                if header is None:
                    break
                kind, length = _HEADER.unpack(header)
                if kind != KIND_JSON:
                    reason = 'unexpected message kind'
                    break
                if length > CLIENT_JSON_MAX:
                    reason = 'oversize command'
                    break
                payload = self._recv_exact(conn, length)
                if payload is None:
                    break
                try:
                    obj = json.loads(payload.decode('utf-8'))
                except ValueError:  # includes JSONDecodeError and UnicodeDecodeError
                    reason = 'bad json'
                    break
                if not isinstance(obj, dict):
                    reason = 'command is not an object'
                    break
                with self._lock:
                    self._commands.append((conn.cid, obj))
        except Exception:
            reason = 'reader error'
            logger.exception('Focus link reader for client %d failed', conn.cid)
        finally:
            self._drop(conn, reason)
            conn.release_user()

    def _sender(self, conn):
        reason = 'sender stopped'
        try:
            while not conn.closed.is_set():
                entry = conn.slot.get(timeout=None)
                if entry is None:
                    break
                what, payload = entry
                kind = KIND_JSON if what == 'json' else KIND_FRAME
                try:
                    conn.sock.sendall(_header(kind, payload))
                    conn.sock.sendall(payload)
                except socket.timeout:
                    reason = 'send timed out'
                    break
                except OSError:
                    reason = 'send failed'
                    break
        except Exception:
            reason = 'sender error'
            logger.exception('Focus link sender for client %d failed', conn.cid)
        finally:
            self._drop(conn, reason)
            conn.release_user()


class LinkClient(object):
    """The web-side end of the link, one per focus page."""

    def __init__(self, path: str, timeout: float = 2.0):
        self.path = path
        self.timeout = timeout
        self._sock = None
        self._selector = None
        self._buf = bytearray()
        self._send_lock = threading.Lock()

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self.path)
        except FileNotFoundError:
            sock.close()
            raise LinkUnavailable('missing')
        except ConnectionRefusedError:
            sock.close()
            raise LinkUnavailable('refused')
        except socket.timeout:
            sock.close()
            raise LinkUnavailable('timeout')
        except OSError:
            sock.close()
            raise LinkUnavailable('unreachable')
        self._sock = sock
        self._selector = selectors.DefaultSelector()
        self._selector.register(sock, selectors.EVENT_READ)
        self._buf = bytearray()

    def send(self, obj: dict) -> None:
        if self._sock is None:
            raise LinkUnavailable('closed')
        payload = _encode_json(obj)
        if len(payload) > CLIENT_JSON_MAX:
            raise ValueError('command larger than the link allows')
        try:
            with self._send_lock:
                self._sock.sendall(_header(KIND_JSON, payload) + payload)
        except socket.timeout:
            raise LinkUnavailable('timeout')
        except OSError:
            raise LinkUnavailable('closed')

    def recv(self, timeout: float | None = None):
        """('json', dict), ('frame', bytes), or None on timeout.

        Raises LinkUnavailable('closed') when the server goes away. A message that is only
        partly received when the timeout passes stays buffered for the next call.
        """
        if self._sock is None:
            raise LinkUnavailable('closed')
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            msg = self._parse()
            if msg is not None:
                return msg
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not self._selector.select(remaining):
                return None
            try:
                chunk = self._sock.recv(1024 * 1024)
            except socket.timeout:
                continue
            except OSError:
                chunk = b''
            if not chunk:
                self.close()
                raise LinkUnavailable('closed')
            self._buf.extend(chunk)

    def _parse(self):
        if len(self._buf) < _HEADER.size:
            return None
        kind, length = _HEADER.unpack_from(self._buf)
        if kind not in (KIND_JSON, KIND_FRAME) or length > SERVER_MSG_MAX:
            logger.warning('Focus link: malformed message from server, closing')
            self.close()
            raise LinkUnavailable('protocol')
        end = _HEADER.size + length
        if len(self._buf) < end:
            return None
        payload = bytes(self._buf[_HEADER.size:end])
        del self._buf[:end]
        if kind == KIND_FRAME:
            return ('frame', payload)
        try:
            obj = json.loads(payload.decode('utf-8'))
        except ValueError:
            logger.warning('Focus link: bad json from server, closing')
            self.close()
            raise LinkUnavailable('protocol')
        return ('json', obj)

    def close(self) -> None:
        if self._selector is not None:
            try:
                self._selector.close()
            except Exception:
                pass
            self._selector = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


def probe(path: str) -> bool:
    """True if a server answers on path within 0.5 s. Never raises."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    except OSError:
        return False
    try:
        s.settimeout(PROBE_TIMEOUT_S)
        s.connect(path)
        return True
    except (OSError, ValueError):
        return False
    finally:
        s.close()
