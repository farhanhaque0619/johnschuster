"""
Rooms: Macs that stay connected and pass the music around.

Every open SyncAudio runs a small agent on port 47801. Two Macs connect once
(one asks, the other accepts) and from then on share a secret key, find each
other whenever both apps are open, and any of them can say "play X's music".
That choice spreads to the whole room; each Mac then starts sharing or
listening by itself.

Messages are single JSON lines over short TCP connections. Everything between
members is signed with the room key, so other people on the network can't
switch your music, and above all can't make your Mac start sharing its audio.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import socket
import threading
import time

ROOM_PORT = 47801
MAGIC = "syncaudio-room/1"
DISCOVER = b"R" + MAGIC.encode()
DISCOVERY_PORTS = [ROOM_PORT]       # tests add more, to run several "Macs" on one machine
ONLINE_FOR = 7.0                    # seconds without contact before a member shows as offline
MAX_AGE = 300.0                     # signed messages older than this are refused (replays)


def _sign(key: bytes, body: dict) -> str:
    return hmac.new(key, json.dumps(body, sort_keys=True).encode(), hashlib.sha256).hexdigest()


def send(ip: str, port: int, message: dict, timeout: float = 2.0):
    """One request, one reply."""
    try:
        with socket.create_connection((ip, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(json.dumps(message).encode() + b"\n")
            data = b""
            while not data.endswith(b"\n") and len(data) < 1 << 20:
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
        return json.loads(data) if data.strip() else None
    except (OSError, ValueError):
        return None


def discover(targets, timeout: float = 1.0, ports=None) -> list:
    """Every SyncAudio (1.1+) answering on the network: [{id, name, version, address, port, room_port, sharing}]."""
    found = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        for target in targets:
            for port in ports or DISCOVERY_PORTS:
                try:
                    s.sendto(DISCOVER, (target, port))
                except OSError:
                    pass
        end = time.monotonic() + timeout
        while (left := end - time.monotonic()) > 0:
            s.settimeout(left)
            try:
                data, addr = s.recvfrom(4096)
            except OSError:
                break
            if data[:1] != b"r":
                continue
            try:
                info = json.loads(data[1:])
            except ValueError:
                continue
            if isinstance(info, dict) and info.get("id"):
                found[info["id"]] = {**info, "address": addr[0]}
    return list(found.values())


class RoomAgent:
    def __init__(self, me: dict, store: dict, save=lambda: None, log=print, on_change=None,
                 port: int = ROOM_PORT, sharing=lambda: False, my_ip=lambda: None):
        """me: {"id", "name", "version", "port" (audio)}. store: persisted {"key", "members"}."""
        self.me, self.store, self.save, self.log = me, store, save, log
        self.on_change = on_change or (lambda: None)
        self.port, self.sharing, self.my_ip = port, sharing, my_ip
        self.lock = threading.RLock()
        self.source = {"id": None, "at": 0.0, "by": None}     # whose music; newest choice wins
        self.seen: dict = {}            # member id -> monotonic time of last contact
        self.versions: dict = {}
        self.requests: dict = {}        # people asking to connect, waiting for Accept
        self.asked: dict = {}           # Macs we asked, waiting for them
        self.running = False

    # ------------------------------------------------------------ state
    @property
    def key(self):
        k = self.store.get("key")
        return bytes.fromhex(k) if k else None

    @property
    def members(self) -> dict:
        return self.store.setdefault("members", {})

    def info(self) -> dict:
        return {**self.me, "room_port": self.port}

    def online(self, member_id: str) -> bool:
        return time.monotonic() - self.seen.get(member_id, -1e9) < ONLINE_FOR

    def status(self) -> dict:
        with self.lock:
            members = [{"id": i, "name": m.get("name"), "ip": m.get("ip"), "online": self.online(i),
                        "version": self.versions.get(i)} for i, m in self.members.items()]
            src = self.source["id"]
            name = self.me["name"] if src == self.me["id"] else (self.members.get(src) or {}).get("name")
            return {
                "id": self.me["id"],
                "members": sorted(members, key=lambda m: (m["name"] or "").lower()),
                "source": {"id": src, "name": name, "by": self.source["by"]},
                "requests": [{"id": i, "name": r.get("name"), "ip": r.get("ip")} for i, r in self.requests.items()],
                "asking": [{"id": i, "name": a.get("name")} for i, a in self.asked.items()],
            }

    # ---------------------------------------------------------- network
    def start(self):
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.tcp.bind(("", self.port))
            self.tcp.listen()
            self.udp.bind(("", self.port))
        except OSError:
            self.tcp.close()
            self.udp.close()
            raise
        self.tcp.settimeout(0.5)
        self.udp.settimeout(0.5)
        self.running = True
        self._threads = [threading.Thread(target=f, daemon=True)
                         for f in (self._accept_loop, self._udp_loop, self._presence_loop)]
        for t in self._threads:
            t.start()
        return self

    def stop(self):
        self.running = False
        for t in getattr(self, "_threads", []):
            t.join(timeout=2)
        for s in (getattr(self, "tcp", None), getattr(self, "udp", None)):
            if s:
                s.close()

    def _udp_loop(self):
        while self.running:
            try:
                data, addr = self.udp.recvfrom(512)
                if data == DISCOVER:
                    reply = {**self.info(), "sharing": bool(self.sharing())}
                    self.udp.sendto(b"r" + json.dumps(reply).encode(), addr)
            except OSError:
                continue

    def _accept_loop(self):
        while self.running:
            try:
                sock, addr = self.tcp.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve, args=(sock, addr[0]), daemon=True).start()

    def _serve(self, sock: socket.socket, ip: str):
        with sock:
            try:
                sock.settimeout(3)
                data = b""
                while not data.endswith(b"\n") and len(data) < 1 << 20:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                reply = self._handle(json.loads(data), ip)
                sock.sendall(json.dumps(reply).encode() + b"\n")
            except (OSError, ValueError, TypeError, AttributeError):
                pass

    # ---------------------------------------------------------- messages
    def _message(self, kind: str, **fields) -> dict:
        body = {"type": kind, "magic": MAGIC, "from": self.info(), "ts": time.time(), **fields}
        key = self.key
        if key:
            body["mac"] = _sign(key, body)
        return body

    def _verified(self, msg: dict, key: bytes | None = None) -> bool:
        key = key or self.key
        if not key or "mac" not in msg:
            return False
        body = {k: v for k, v in msg.items() if k != "mac"}
        return (hmac.compare_digest(_sign(key, body), str(msg["mac"]))
                and abs(time.time() - float(msg.get("ts", 0))) < MAX_AGE)

    def _roster(self) -> dict:
        roster = {i: dict(m) for i, m in self.members.items()}
        roster[self.me["id"]] = {"name": self.me["name"], "ip": self.my_ip(),
                                 "port": self.me["port"], "room_port": self.port}
        return roster

    def _sync_message(self) -> dict:
        return self._message("sync", source=self.source, members=self._roster())

    def _touch(self, sender: dict, ip: str):
        """Fresh contact with a member: remember where it is now."""
        sid = sender["id"]
        member = self.members.get(sid)
        if member is None:
            return
        update = {"name": sender.get("name", member.get("name")), "ip": ip,
                  "port": sender.get("port", member.get("port")), "room_port": sender.get("room_port", member.get("room_port"))}
        moved = any(member.get(k) != v for k, v in update.items())
        member.update(update)
        self.seen[sid] = time.monotonic()
        self.versions[sid] = sender.get("version")
        if moved:
            self.save()
            self.on_change()

    def _merge(self, msg: dict):
        """Take on what another member knows: new members, a newer music choice."""
        changed = False
        for mid, info in (msg.get("members") or {}).items():
            if mid != self.me["id"] and mid not in self.members and isinstance(info, dict):
                self.members[mid] = {k: info.get(k) for k in ("name", "ip", "port", "room_port")}
                changed = True
        if changed:
            self.save()
        source = msg.get("source") or {}
        if float(source.get("at", 0)) > self.source["at"]:
            self.source = {"id": source.get("id"), "at": float(source["at"]), "by": source.get("by")}
            changed = True
        if changed:
            self.on_change()

    def _handle(self, msg: dict, ip: str) -> dict:
        kind, sender = msg.get("type"), msg.get("from") or {}
        sid = sender.get("id")
        if msg.get("magic") != MAGIC or not sid or sid == self.me["id"]:
            return {"ok": False}
        with self.lock:
            if kind == "connect":           # someone asks to join our room: the person here decides
                self.requests[sid] = {**sender, "ip": ip, "at": time.time()}
                self.on_change()
                return {"ok": True, "status": "pending", "from": self.info()}
            if kind == "welcome":           # our request was accepted: here's the key and the room
                if sid not in self.asked or not isinstance(msg.get("key"), str):
                    return {"ok": False}
                self.asked.pop(sid)
                old_key, old_members = self.key, {i: dict(m) for i, m in self.members.items()}
                self.store["key"] = msg["key"]
                self.members[sid] = {}
                self._touch(sender, ip)
                self._merge(msg)
                self.save()
                if old_key and old_members:     # bring the Macs we were already with along
                    rekey = {"type": "rekey", "magic": MAGIC, "from": self.info(), "ts": time.time(),
                             "key": msg["key"], "members": self._roster(), "source": self.source}
                    rekey["mac"] = _sign(old_key, rekey)
                    for m in old_members.values():
                        threading.Thread(target=send, args=(m.get("ip"), m.get("room_port") or ROOM_PORT, rekey),
                                         daemon=True).start()
                self.log(f"Connected with {sender.get('name')}.")
                self.on_change()
                return {"ok": True}
            if not self._verified(msg):
                return {"ok": False, "error": "not in this room"}
            if kind == "rekey":
                self.store["key"] = msg["key"]
                self._merge(msg)
                self.save()
                return {"ok": True}
            if sid not in self.members:
                self.members[sid] = {}
            if kind == "leave":
                self.members.pop(sid, None)
                self.save()
                self.on_change()
                return {"ok": True}
            if kind == "sync":
                self._touch(sender, ip)
                self._merge(msg)
                return self._sync_message()
        return {"ok": False}

    # ----------------------------------------------------------- actions
    def request_connect(self, ip: str, room_port: int = ROOM_PORT) -> dict:
        """Ask the Mac at ip to let us into its room."""
        reply = send(ip, room_port, {"type": "connect", "magic": MAGIC, "from": self.info(), "ts": time.time()})
        if not reply or not reply.get("ok"):
            return {"ok": False, "message": "That Mac didn't answer. Is SyncAudio 1.1 or newer open on it?"}
        target = reply.get("from") or {}
        with self.lock:
            self.asked[target.get("id")] = {**target, "ip": ip}
        self.on_change()
        return {"ok": True, "name": target.get("name")}

    def accept(self, sid: str) -> bool:
        with self.lock:
            req = self.requests.pop(sid, None)
            if not req:
                return False
            if not self.key:
                self.store["key"] = secrets.token_hex(32)
            self.members[sid] = {k: req.get(k) for k in ("name", "ip", "port", "room_port")}
            self.seen[sid] = time.monotonic()
            self.save()
            welcome = {"type": "welcome", "magic": MAGIC, "from": self.info(), "ts": time.time(),
                       "key": self.store["key"], "members": self._roster(), "source": self.source}
        self.on_change()
        ok = send(req["ip"], req.get("room_port") or ROOM_PORT, welcome) is not None
        self.log(f"Connected with {req.get('name')}." if ok else f"! {req.get('name')} didn't answer.")
        return ok

    def decline(self, sid: str):
        with self.lock:
            self.requests.pop(sid, None)
        self.on_change()

    def forget(self, sid: str):
        """Leave a member's company: both sides forget each other."""
        with self.lock:
            member = self.members.pop(sid, None)
            message = self._message("leave")
            if self.source["id"] == sid:
                self.source = {"id": None, "at": time.time(), "by": self.me["name"]}
            self.save()
        if member:
            send(member.get("ip"), member.get("room_port") or ROOM_PORT, message)
        self.on_change()

    def set_source(self, sid):
        """Choose whose music everyone hears (None: nobody's)."""
        with self.lock:
            self.source = {"id": sid, "at": time.time(), "by": self.me["name"]}
        self.on_change()
        self._broadcast()

    def saw(self, info: dict):
        """Discovery found a Mac; if it's a member, note where it is now."""
        with self.lock:
            member = self.members.get(info.get("id"))
            if member and member.get("ip") != info.get("address"):
                member.update(ip=info.get("address"), port=info.get("port", member.get("port")),
                              room_port=info.get("room_port", member.get("room_port")))
                self.save()

    # ---------------------------------------------------------- keeping in touch
    def _exchange(self, mid: str):
        with self.lock:
            member = self.members.get(mid)
            message = self._sync_message() if member else None
        if not member or not member.get("ip"):
            return
        reply = send(member["ip"], member.get("room_port") or ROOM_PORT, message)
        if reply and self._verified(reply) and (reply.get("from") or {}).get("id") == mid:
            with self.lock:
                self._touch(reply["from"], member["ip"])
                self._merge(reply)

    def _broadcast(self):
        for mid in list(self.members):
            threading.Thread(target=self._exchange, args=(mid,), daemon=True).start()

    def _presence_loop(self):
        while self.running:
            was = {mid: self.online(mid) for mid in self.members}
            self._broadcast()
            time.sleep(2.0)
            if any(self.online(mid) != up for mid, up in was.items()):
                self.on_change()
