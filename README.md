# syncaudio

Play whatever one Mac is playing (Spotify, YouTube, a game, anything) out of
**two or more MacBooks' speakers at the same time, in sync**.

```
 your Mac (host)                                   friend's Mac
 Spotify / YouTube / anything
        │  (sound output → BlackHole)
        ▼
 syncaudio host ───── Wi-Fi: audio + clock sync ────► syncaudio join
        │                                                   │
        ▼                                                   ▼
 speakers play sample n at time T  ◄──── same T ────►  speakers play sample n at time T
```

The host captures its system audio through BlackHole (a free virtual audio
cable). Each block of audio is stamped with the exact moment **T** it should
come out of the speakers. Every Mac, the host included, waits until T before
playing it, so both speakers play the same sample together. Neither Mac runs
ahead of the other.

* The Macs keep their clocks matched with constant pings, trusting only the
  fastest round trips. In simulated tests with drifting clocks and jittery
  Wi-Fi, the two Macs stay within about **1 ms** of each other. For scale,
  sound takes about 3 ms to cross one metre of room.
* CoreAudio reports each speaker's own output delay, and each Mac corrects
  for it.
* Small clock drift is fixed by adding or skipping a single sample now and
  then, which you can't hear. A Wi-Fi hiccup gives a moment of silence, but
  the two Macs never drift apart.
* Total delay from your Mac to the speakers is **120 ms** by default
  (adjustable, see below).

## Setup (once)

You need [Homebrew](https://brew.sh) on both Macs, and both Macs on the same
Wi-Fi (not a guest network).

Get this folder onto both Macs (`git clone`, or AirDrop it), then in Terminal:

```bash
# On YOUR Mac (the one that plays the music):
./setup.sh host

# On your FRIEND's Mac:
./setup.sh
```

`setup.sh host` also installs **BlackHole** (it asks for your password) and
**SwitchAudioSource**. If BlackHole doesn't show up afterwards, restart the Mac.

## Use

```bash
# Your Mac
./syncaudio host

# Friend's Mac (finds your Mac automatically)
./syncaudio join
```

Then play anything on your Mac. While hosting, your Mac's sound output is
switched to BlackHole, and it switches back when you quit.

The first time you run it, macOS asks a couple of things. Say **Allow** to both:
* *"Terminal would like to access the microphone"* (on the host). BlackHole
  counts as a microphone, and without this you stream silence.
* *"Accept incoming network connections"* for Python (on the host).

If auto-discovery doesn't find the host (some routers block it), use the IP
address that `./syncaudio host` prints: `./syncaudio join 192.168.1.23`.

### Check the sync by ear (recommended first time)

```bash
./syncaudio host --click     # host: plays a metronome instead of your Mac's audio
./syncaudio join             # friend
```

Put the Macs side by side. If they're in sync you hear **one crisp click**.
If you hear a "flam" (a doubled *t-tick*), press `]` or `[` on the friend's
Mac to nudge it later or earlier by 1 ms until the click is clean. You can also
pass the value at startup next time, for example `./syncaudio join --offset 3`.

### Keys (in the Terminal window)

| key       | does                                                   |
|-----------|--------------------------------------------------------|
| `+` / `-` | this Mac's volume                                      |
| `]` / `[` | play this Mac 1 ms later / earlier (`}` `{` = 10 ms)   |
| `c`       | cycle: both channels → left only → right only          |
| `q`       | quit                                                   |

The host's normal volume keys control the shared stream, so they change the
volume for everyone. The friend's volume keys only change their own Mac.

**Stereo trick:** put the Macs a few metres apart, press `c` once on the
host (left only) and twice on the friend's Mac (right only). Now you have real
stereo speakers.

## Tuning the delay

`--delay` (host only) is the time from your Mac playing a sound to it coming
out of the speakers.

```bash
./syncaudio host --delay 80     # snappier, needs good Wi-Fi
./syncaudio host --delay 200    # rock solid on busy / weak Wi-Fi
```

The friend's status line shows **headroom**, which is how early the audio
arrives before it's needed. If headroom drops near 0 or you get dropouts,
raise `--delay`. If headroom stays above 50 ms, you can lower it.

With **YouTube or other video**, the sound is now `--delay` ms behind the
picture on your screen. 120 ms is barely noticeable. If lips look off, try
`--delay 70` or `80`.

## Troubleshooting

* **Friend can't find the host:** use `./syncaudio join <ip>`. Make sure both
  Macs are on the same network, not a guest or "isolated" Wi-Fi, which blocks
  devices from talking to each other.
* **Host shows "silent":** play something. If something *is* playing, allow
  Terminal in System Settings → Privacy & Security → Microphone, then restart
  `./syncaudio host`.
* **Wrong speaker:** `./syncaudio devices` lists audio devices, and you pick
  one with `--output "name"`. Built-in or wired speakers sync best.
  Bluetooth/AirPods report their delay less precisely, so use `[` `]` to fix
  them by ear.
* **No sound on the host after a crash:** set the output back in Control Center
  → Sound.
* More than two Macs work too: run `./syncaudio join` on each.

## Running the tests

```bash
python3 -m unittest discover tests
```

The tests simulate two Macs whose clocks and sound cards drift apart over a
jittery network, then run real host/friend sessions over the network with
simulated speakers. They check that the same click comes out of both within
2 ms. They need `numpy` but no audio hardware.
