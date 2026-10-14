"""
Finding Macs that are sharing, on networks that make it hard.

Three ways, used together:
  * Bonjour (macOS's dns-sd tool). The sharing Mac puts its network address in
    the announcement. (It can also use the AirDrop link, but doesn't by default:
    that link makes Wi-Fi hop channels and fails when it's turned off.)
  * A sweep of the whole local subnet (the real netmask, not just a /24),
    a chunk per scan, for networks that block broadcasts and Bonjour.
  * Addresses that worked before (remembered by the app).
"""
from __future__ import annotations

import ipaddress
import re
import shutil
import subprocess
import threading

SERVICE = "_syncaudio._tcp"            # a Mac that's sharing
ROOM_SERVICE = "_syncaudio-room._tcp"  # any open SyncAudio 1.1+ (rooms)
P2P = ["-includep2p", "-includeAWDL"]    # also over AWDL, the AirDrop link (off by default:
                                          # it makes Wi-Fi hop channels, which Bluetooth headphones
                                          # sharing the radio hear, and finds nothing when AWDL is off)
SWEEP_CHUNK = 512                         # addresses per scan


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
    m = re.match(r"^\S+\s+(Add|Rmv)\s+\S+\s+\d+\s+\S+\s+\S+\s+(.+?)\s*$", line)
    return (m.group(1), m.group(2)) if m else None


def parse_resolve(text: str):
    """The announced address (ip=...) and port from `dns-sd -L` output."""
    port = re.search(r"can be reached at \S+?:(\d+)", text)
    ip = re.search(r"\bip=(\d+\.\d+\.\d+\.\d+)", text)
    return (ip.group(1) if ip else None, int(port.group(1)) if port else None)


class Advertiser:
    """Announces this Mac while it shares. Over the AirDrop link only for the
    first few minutes: that link makes Wi-Fi hop channels, which costs a little
    latency, so it stops once people have had time to find this Mac."""

    P2P_SECONDS = 0

    def __init__(self, name: str, port: int, ip: str, service: str = SERVICE, p2p_seconds: float = P2P_SECONDS):
        self.name, self.port, self.ip, self.service = name, port, ip, service
        self.p2p_seconds = p2p_seconds
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
        self.set_p2p(self.p2p_seconds > 0)
        if self.p2p_seconds > 0 and not self._stop.wait(self.p2p_seconds):
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
