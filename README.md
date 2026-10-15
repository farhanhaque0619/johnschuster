# SyncAudio

Play whatever one Mac is playing (Spotify, YouTube, a game, anything) out of
**any number of Macs at the same time, in sync**. Put two MacBooks in a room
and they sound like one bigger speaker, or like a stereo pair.

<p>
  <img src="docs/sharing.png" width="300" alt="The Mac that shares: a level meter and a list of Macs that are in sync">
  &nbsp;
  <img src="docs/listening.png" width="300" alt="A Mac that joined: in sync, with volume and sync controls">
</p>

## Get it

1. **Download [SyncAudio.dmg](https://github.com/farhanhaque0619/johnschuster/releases/latest/download/SyncAudio.dmg)**.
   It's for Macs with Apple silicon (M1 or newer) running macOS 14 Sonoma or later.
2. Open the `.dmg` and drag **SyncAudio** onto **Applications**.
3. Open SyncAudio from Applications.
   * **"Apple could not verify SyncAudio…"**: click **Done**, open **System Settings →
     Privacy & Security**, scroll down, click **Open Anyway** and confirm. You only do
     this once. (It appears because the app isn't from the App Store.)
4. When it asks to **find devices on your local network**, click **Allow**.

Every Mac that plays music needs SyncAudio installed, and all of them must be on
the **same Wi-Fi**.

### Updating

SyncAudio updates itself. When a new version is out, a banner with **Update now**
appears. Click it and the app restarts as the new version a few seconds later.
macOS may ask for the microphone and local-network permissions again; click **Allow**.

**Installing or updating from Terminal** (handy for older versions without the
updater, and it skips the "can't verify" prompt):

```bash
curl -fsSL https://github.com/farhanhaque0619/johnschuster/releases/latest/download/install.sh | bash
```

## What's new in 2.0

SyncAudio 2.0 rebuilds the part that plays the sound, to get rid of the static
some people heard on AirPods.

* **The part that feeds your speakers or headphones is now a tiny piece of
  native code** that never waits for the rest of the app. It always asks macOS
  for a big playback buffer, and it times every sample by macOS's own clock. If
  macOS ever drops a bit of audio, 2.0 counts it and stays exactly in time
  instead of slowly falling behind.
* **Both Macs must be on 2.0.** 1.x and 2.0 can't play together. They still see
  each other, and the app tells you which Mac needs updating.
* **The delay is longer with AirPods (about 0.6 s).** Play and pause in Spotify
  feel a little slower to respond. That's on purpose: it's what keeps Wi-Fi
  hiccups from ever being heard. It may shrink a bit after a few quiet minutes,
  and when it has to change, SyncAudio waits for a quiet moment, like the gap
  between songs.
* **New: Settings → Test my headphones.** It plays a quiet tone for 30 seconds:
  the first 10 the old way (which may crackle), the next 20 with 2.0. Then it
  tells you whether macOS played every bit of it. Run it once on each Mac with
  your headphones in.
* Each Mac remembers its **Sync nudge** separately for every speaker and pair of
  headphones. Nudges from 1.x start again at 0: they made up for 1.x's own timing,
  which 2.0 doesn't have.

## Use it

### Connect your Macs once, then switch whose music plays

1. Open SyncAudio on both Macs. Each shows the other under **Macs nearby**.
2. On one Mac click **Connect**; on the other click **Accept**.
3. Both now show **Connected Macs**. Click **Play my music** (or **Play theirs**) on
   either Mac to choose whose music everyone hears. Switch any time, from any
   Mac: the Mac you pick starts sharing and the others start listening by
   themselves. **Stop the music** stops it for everyone.

Connected Macs remember each other and reconnect whenever both apps are open.
Only Macs you've accepted can switch your music. ✕ next to a Mac disconnects it.

### Or share without connecting

**On the Mac playing the music:** click **Start sharing**.

* The first time, SyncAudio offers to install **BlackHole**, a free audio
  driver that lets it hear what your Mac plays. Click **Install BlackHole** and
  type your Mac password. Only the sharing Mac needs it.
* Click **Allow** when asked about the microphone. SyncAudio listens to
  BlackHole, never to your real microphone.
* Play anything. The bar moves when sound is coming through.

**On every other Mac:** open SyncAudio. The sharing Mac appears under
**Macs nearby**; click **Just listen** (or **Join** for older versions). That's it. Add as many Macs as
you like, and they all stay in sync.

**Controls (each Mac has its own):**

| Control       | What it does                                                                 |
|---------------|------------------------------------------------------------------------------|
| Volume: Everyone | Changes the volume on **every** Mac at once. Anyone, sharing or listening, can move it, and everyone sees who changed it last. |
| Volume: This Mac | Only this Mac. (On a listening Mac, the keyboard volume keys work too.) |
| Sync nudge    | If you hear an echo or a double beat between Macs, nudge one until they sound like a single speaker. |
| Both / Left / Right | Put two Macs apart, set one to Left and the other to Right: real stereo. |
| Delay (sharing Mac) | **Auto** keeps it as low as the Wi-Fi allows and raises it if anyone's audio arrives late. |
| Play test clicks | Plays a metronome to check the sync by ear: one crisp click = in sync. |

When you click **Stop**, your Mac's sound goes back to its speakers.

### AirPods and headphones

They work, on any Mac, sharing or listening. Put them in whenever you like:
SyncAudio notices when macOS switches the sound to them and moves the music
over within a couple of seconds. On the sharing Mac it also keeps the sharing
going, so everyone else still hears it.

* SyncAudio never uses the AirPods' microphone, so they stay in full-quality
  music mode. (Joining a call on them switches them to call mode, which sounds
  worse. That's macOS, not SyncAudio; the app tells you when it happens.)
* AirPods hold on to quite a lot of audio before playing it. SyncAudio measures
  how much on every Mac, remembers it, and starts with enough delay to cover it
  (about 0.6 s), so leave **Delay** on **Auto**.
* macOS's figure for how late Bluetooth plays isn't perfect. If someone on
  AirPods hears things slightly early or late compared to speakers in the room,
  use **Sync nudge** on their Mac.

## If something's off

* **The other Mac doesn't show up under Join.** Big networks (schools, offices,
  apartment buildings) often hide Macs from each other. Type the address the
  sharing Mac shows (it has a **Copy** button) into the box under the list, and
  click **Test** to check it. After you've played together once, each Mac remembers
  the other and finds it by itself next time. Also check **System Settings →
  Privacy & Security → Local Network** has SyncAudio turned on.
* **"Can't reach …" when joining.** The app says why underneath: wrong address,
  the network keeping devices apart, the Mac not sharing, or macOS blocking
  local-network access.
* **"… needs updating" or "Different versions".** SyncAudio 2.0 can't play
  together with 1.x. Update the other Mac (**Settings → Check for updates**, or
  the Terminal line under [Updating](#updating)).
* **The bar doesn't move while music plays** (listeners hear silence; their app says
  so). The sharing Mac's sound output must be **BlackHole 2ch**, not "Multi-Output
  Device" or anything else. SyncAudio sets this itself and puts it back if it changes.
  Also turn on SyncAudio in **System Settings → Privacy & Security → Microphone**
  (macOS may ask again after an update), then stop and start sharing.
* **Others can't join you, but you can join them.** Your Mac's firewall is probably
  blocking SyncAudio. The sharing screen shows **Allow SyncAudio** when that's the case.
* **Audio drops out.** With delay on Auto, SyncAudio raises the delay within a few
  seconds. If it keeps happening, move closer to the router or use 5 GHz Wi-Fi.
  Macs also briefly pause Wi-Fi to scan for AirDrop. Running
  `sudo ifconfig awdl0 down` in Terminal on each Mac turns that off until restart,
  which often fixes repeated dropouts.
* **Static on AirPods or other Bluetooth headphones.** First make sure both Macs
  have 2.0. Then stop sharing or listening and run **Settings → Test my headphones**
  (30 seconds, a quiet tone). If it says **macOS is dropping audio on these
  headphones**, the static is between macOS and the headphones, not in SyncAudio.
  Try: quit apps that use the mic (Zoom, FaceTime), turn Bluetooth off and on,
  unpair other Bluetooth devices nearby, and check whether the same song from the
  Music app also crackles. Then send us diagnostics.
* **Still static, and the test says it's clean.** A Mac's Wi-Fi and Bluetooth share
  one radio, so a busy Wi-Fi connection makes Bluetooth headphones drop audio.
  Check which Wi-Fi band the Mac is on: hold **Option** and click the Wi-Fi icon in
  the menu bar. **2.4 GHz** (channels 1-14) is the bad one; join the 5 GHz version
  of the network if there is one (often the same name with "5G"), or sit closer to
  the router. Copy diagnostics shows the channel too.
* **Play and pause feel slow with AirPods.** That's the longer delay 2.0 keeps with
  Bluetooth headphones (about 0.6 s), so a Wi-Fi hiccup is never heard. It can
  shrink a little after a few calm minutes.
* **Still stuck?** Open **Settings → Copy diagnostics** and paste the result to
  whoever is helping you. That's a short summary; the full report (versions, network
  details, audio devices, the headphone test and the last few minutes of sync
  measurements) is saved as **SyncAudio diagnostics.txt** on your Desktop.
* **No sound on the sharing Mac after a crash.** Open SyncAudio again and it puts the
  sound back, or pick your speakers in **Control Center → Sound**.

## How it works

The sharing Mac captures its audio through BlackHole and stamps every 10 ms
block with the exact moment it should come out of the speakers, compressed with
Opus (about 100 kbit/s instead of 1.5 Mbit/s). Every Mac,
including the sharing one, holds each block until that moment. The Macs
match clocks by pinging each other constantly and trusting only the fastest
round trips.

On each Mac, a small piece of C code hands the sound to macOS. It runs on
macOS's own audio thread and never waits for the rest of the app, which keeps it
filled up to a second ahead. Every sample is timed by macOS's clock for that
speaker, including its own delay, and the music is played a few hundredths of a
percent faster or slower to stay exactly on time, which you can't hear.

In simulated tests with drifting clocks and jittery Wi-Fi, the Macs stay
within about 1 ms of each other. For scale, sound takes about 3 ms to cross
a metre of room.

---

## For developers

The app is `app.py` (a small local web server plus a native window) around
the engine in `syncaudio.py`, with the interface in `ui/index.html`. The audio
itself goes through `audio_engine.py` and libsacore (`native/sacore.c`), the C
code that runs on CoreAudio's IO threads; `sacore.py` loads it.

**Run from source** (Python 3.12, and Apple's command line tools for the C part:
`xcode-select --install`):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pywebview
bash native/build.sh                 # builds libsacore.dylib; nothing plays without it
.venv/bin/python app.py              # or: app.py --browser
```

**Build the .dmg:** pushing to `main` or `claude/dreamy-johnson-s8n24w` runs
`.github/workflows/build-mac.yml`. On an Apple silicon Mac it runs the
tests, builds `SyncAudio.app` with PyInstaller (`packaging/SyncAudio.spec`),
self-tests it, packs `SyncAudio.dmg`, and publishes it on the
[download page](https://github.com/farhanhaque0619/johnschuster/releases/latest).
The app is ad-hoc signed, not notarized, which is why macOS asks you to confirm
the first launch. With an Apple Developer account it could be signed and
notarized so that prompt goes away.

**Tests** (no audio hardware needed; Linux works too):

```bash
python3 tests/run_ci.py         # builds libsacore (native/build.sh) first
```

They simulate Macs whose clocks and sound cards drift apart over jittery Wi-Fi,
play SyncAudio's real C audio callbacks against simulated devices (in virtual
time, and in real time on a fake audio thread), run real sessions with three
Macs over the network, and drive the app's API.

Before a release, the workflow also plays through real CoreAudio on a CI Mac
(`tests/coreaudio/run_all.py`): BlackHole loops the sound back, a separate
process records it, and every recorded sample is checked against when it should
have played, under deliberate load. The old 1.1.6 engine must fail that test
(or it couldn't see the bug); 2.0 must pass it. To try those scripts anywhere,
on fake loopback devices: `python3 tests/coreaudio/run_all.py --fake --twins yes --scale 0.25`.

### Terminal version

The same engine works from the Terminal, without the app:

```bash
./setup.sh host                 # on the sharing Mac (installs BlackHole via Homebrew)
./setup.sh                      # on the others (both build the audio core: needs xcode-select --install)
./syncaudio host                # share (add --click for the metronome)
./syncaudio join [address]      # listen; finds the host by itself
./syncaudio selftest            # "Test my headphones" on the current output
```

Keys while running: `+`/`-` volume, `[` `]` nudge 1 ms (`{` `}` 10 ms),
`c` cycles both/left/right, `q` quits. `./syncaudio host --delay 250` fixes the
delay instead of choosing it automatically.
