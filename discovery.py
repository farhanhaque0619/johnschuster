"""
Finding Macs that are sharing, on networks that make it hard.

Three ways, used together:
  * Bonjour (macOS's dns-sd tool). The sharing Mac puts its network address in
    the announcement.
  * A sweep of the whole local subnet (the real netmask, not just a /24),
    a chunk per scan, for networks that block broadcasts and Bonjour.
  * Addresses that worked before (remembered by the app).

And when the Wi-Fi keeps devices apart (many school, office and guest networks
show Macs to each other but drop every connection between them): the Macs' own
peer-to-peer Wi-Fi link, the one AirDrop uses (p2p_hosts). It only comes up while
someone asks for it - the sharing Mac announces itself on it while nobody is
connected - because it makes Wi-Fi hop channels.
"""
from __future__ import annotations

import ipaddress
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

SERVICE = "_syncaudio._tcp"            # a Mac that's sharing
ROOM_SERVICE = "_syncaudio-room._tcp"  # any open SyncAudio 1.1+ (rooms)
P2P = ["-includep2p", "-includeAWDL"]    # also over AWDL, the AirDrop link (off by default:
                                          # it makes Wi-Fi hop channels, which Bluetooth headphones
                                          # sharing the radio hear, and finds nothing when AWDL is off)
SWEEP_CHUNK = 512                         # addresses per scan
SO_RECV_ANYIF = 0x1104                    # macOS: let a socket use the peer-to-peer link (sys/socket.h)


def allow_p2p(sock) -> bool:
    """Let this socket send and receive over the AirDrop link (awdl0), which macOS keeps sockets
    off by default. (A private option, but the only one BSD sockets have; harmless elsewhere.)"""
    if sys.platform != "darwin":
        return False
    try:
        sock.setsockopt(socket.SOL_SOCKET, SO_RECV_ANYIF, 1)
        return True
    except OSError:
        return False


def is_p2p_interface(name: str | None) -> bool:
    """The AirDrop link's interfaces: awdl0 (and llw0, its low-latency twin)."""
    return bool(name) and name.startswith(("awdl", "llw"))


def interface_for(ip: str, ifconfig_text: str | None = None):
    """(network, broadcast) of the interface holding `ip`, from ifconfig."""
    if ifconfig_text is None:
        try:
            ifconfig_text = subprocess.run(["ifconfig"], capture_output=True, text=True, timeout=2).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    for m in re.finditer(r"inet (\d+\.\d+\.\d+\.\d+) netmask (0x[0-9a-fA-F]+|\d+\.\d+\.\d+\.\d+)"
                         r"(?: broadcast (\d+\.\d+\.\d+\.\d+))?", ifconfig_text):
        if m.group(1) != ip:
            continue
        mask = m.group(2)
        mask = str(ipaddress.IPv4Address(int(mask, 16))) if mask.startswith("0x") else mask
        net = ipaddress.IPv4Network(f"{ip}/{mask}", strict=False)
        return net, m.group(3) or str(net.broadcast_address)
    return None


class Sweep:
    """Walks the local subnet a chunk at a time, nearest addresses first."""

    def __init__(self, ip: str, network: ipaddress.IPv4Network, chunk: int = SWEEP_CHUNK):
        me = int(ipaddress.IPv4Address(ip))
        # Campus networks can be /15 (131k addresses); bigger ones are limited to the /15 around us.
        net = network if network.num_addresses <= 1 << 17 else ipaddress.IPv4Network(f"{ip}/15", strict=False)
        hosts = [int(h) for h in net.hosts()]
        hosts.sort(key=lambda h: abs(h - me))       # neighbours first: DHCP tends to cluster
        self.order = [str(ipaddress.IPv4Address(h)) for h in hosts if h != me]
        self.chunk, self.pos = chunk, 0

    def next_targets(self) -> list:
        if not self.order:
            return []
        part = [self.order[(self.pos + i) % len(self.order)] for i in range(min(self.chunk, len(self.order)))]
        self.pos = (self.pos + self.chunk) % len(self.order)
        return part


# ------------------------------------------------------------------ Bonjour
def dns_sd() -> str | None:
    return shutil.which("dns-sd")


def parse_browse_line(line: str):
    """'12:00:00.000  Add  3  5 local.  _syncaudio._tcp.  Den' -> ("Add", "Den")."""
    m = re.match(r"^\s*\S+\s+(Add|Rmv)\s+\S+\s+\d+\s+\S+\s+\S+\s+(.+?)\s*$", line)
    return (m.group(1), m.group(2)) if m else None


def parse_resolve(text: str):
    """The announced address (ip=...) and port from `dns-sd -L` output."""
    port = re.search(r"can be reached at \S+?:(\d+)", text)
    ip = re.search(r"\bip=(\d+\.\d+\.\d+\.\d+)", text)
    return (ip.group(1) if ip else None, int(port.group(1)) if port else None)


def parse_targets(text: str) -> list:
    """Every (host, port, interface index) a `dns-sd -L` resolve reached the service on: one per
    interface (Wi-Fi, the AirDrop link...)."""
    return [(m.group(1), int(m.group(2)), int(m.group(3)))
            for m in re.finditer(r"can be reached at (\S+?):(\d+) \(interface (\d+)\)", text)]


def parse_addr6(text: str) -> list:
    """IPv6 addresses from `dns-sd -G v6` output, as Python can connect to them ("fe80::1%awdl0")."""
    out = []
    for m in re.finditer(r"\b([0-9A-Fa-f]{4}(?::[0-9A-Fa-f]{4}){7})%(\S+)", text):
        try:
            addr = ipaddress.IPv6Address(m.group(1))
        except ValueError:
            continue
        out.append(f"{addr.compressed}%{m.group(2)}" if addr.is_link_local else addr.compressed)
    return out


def _read(cmd: list, seconds: float, done=None) -> str:
    """dns-sd's output for up to `seconds` (it never exits by itself), or until done(text) says enough."""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError:
        return ""
    timer = threading.Timer(seconds, proc.terminate)
    timer.daemon = True
    timer.start()
    text = ""
    try:
        for line in proc.stdout:
            text += line
            if done and done(text):
                break
    finally:
        timer.cancel()
        proc.terminate()
    return text


def p2p_hosts(service: str = SERVICE, seconds: float = 3.0, names=None) -> list:
    """Sharing Macs reachable over the AirDrop link: [{"name", "ip" (their Wi-Fi address), "address"
    (on the link, "fe80::...%awdl0"), "port"}]. Asking brings the link up on this Mac for a moment.
    `names`: only look these up (when we know whom we want)."""
    tool = dns_sd()
    if not tool:
        return []
    wanted = set(names) if names else None
    found = []
    if wanted is None:
        browse = _read([tool, *P2P, "-B", service, "local"], seconds)
        wanted = {p[1] for p in map(parse_browse_line, browse.splitlines()) if p and p[0] == "Add"}
    for name in sorted(wanted):
        text = _read([tool, *P2P, "-L", name, service, "local"], seconds)
        ip, _ = parse_resolve(text)
        for host, port, index in parse_targets(text):
            try:
                ifname = socket.if_indextoname(index)
            except OSError:
                continue
            if not is_p2p_interface(ifname):
                continue
            out = _read([tool, *P2P, "-G", "v6", host], seconds, lambda t: "%" + ifname in t)
            for address in parse_addr6(out):
                if address.endswith("%" + ifname):
                    found.append({"name": name, "ip": ip, "address": address, "port": port})
                    break
            break
    return found


class Advertiser:
    """Announces this Mac while it shares: on the Wi-Fi network, and (p2p) over the AirDrop link too,
    for Wi-Fi that keeps devices apart. The sharing session switches the link on and off (set_p2p):
    it makes Wi-Fi hop channels, so only while it may be needed. p2p_seconds: on for that long at
    the start, then off."""

    P2P_SECONDS = 0

    def __init__(self, name: str, port: int, ip: str, service: str = SERVICE, p2p_seconds: float = P2P_SECONDS,
                 p2p: bool = False):
        self.name, self.port, self.ip, self.service = name, port, ip, service
        self.p2p_seconds = p2p_seconds
        self.p2p_start = p2p
        self.proc = None
        self.p2p = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        if dns_sd():
            threading.Thread(target=self._run, daemon=True).start()
        return self

    def set_p2p(self, on: bool):
        """Announce over the AirDrop link too (on) or only on the Wi-Fi network (off)."""
        with self._lock:
            if self._stop.is_set() or on == self.p2p or not dns_sd():
                return
            if self.proc:
                self.proc.terminate()
            cmd = [dns_sd(), *(P2P if on else []), "-R", self.name, self.service, "local", str(self.port),
                   f"ip={self.ip}", "v=1"]
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.p2p = on

    def _run(self):
        self.set_p2p(self.p2p_start or self.p2p_seconds > 0)
        if not self.p2p_start and self.p2p_seconds > 0 and not self._stop.wait(self.p2p_seconds):
            self.set_p2p(False)

    def stop(self):
        with self._lock:
            self._stop.set()
            if self.proc:
                self.proc.terminate()


class Browser:
    """Keeps a live list of sharing Macs found with Bonjour: {name: ip}."""

    def __init__(self, p2p: bool = False, service: str = SERVICE):
        self.p2p, self.service = p2p, service
        self.found: dict = {}
        self.proc = None
        self._stop = threading.Event()

    def start(self):
        if dns_sd():
            threading.Thread(target=self._run, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()
        if self.proc:
            self.proc.terminate()

    def _run(self):
        flags = P2P if self.p2p else []
        self.proc = subprocess.Popen([dns_sd(), *flags, "-B", self.service, "local"], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        for line in self.proc.stdout:
            if self._stop.is_set():
                break
            parsed = parse_browse_line(line)
            if not parsed:
                continue
            op, name = parsed
            if op == "Rmv":
                self.found.pop(name, None)
            elif name not in self.found:
                threading.Thread(target=self._resolve, args=(name, flags), daemon=True).start()

    def _resolve(self, name: str, flags: list):
        try:
            proc = subprocess.Popen([dns_sd(), *flags, "-L", name, self.service, "local"], stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        timer = threading.Timer(5, proc.terminate)      # dns-sd never exits by itself
        timer.daemon = True
        timer.start()
        text = ""
        for line in proc.stdout:
            text += line
            if "ip=" in text:
                break
        proc.terminate()
        timer.cancel()
        ip, _ = parse_resolve(text)
        if ip and not self._stop.is_set():
            self.found[name] = ip
