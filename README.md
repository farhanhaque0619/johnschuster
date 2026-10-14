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
  worse. That's macOS, not SyncAudio.)
* AirPods hold on to quite a lot of audio before playing it. SyncAudio measures
  how much on every Mac and raises the delay to cover it automatically, so leave
  **Delay** on **Auto**.
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
* **Static on Bluetooth headphones (AirPods, Beats).** A Mac's Wi-Fi and Bluetooth
  share one radio, so a busy Wi-Fi connection makes Bluetooth headphones drop audio.
  Since 1.1.4 SyncAudio compresses the stream to about a fifteenth of its old size. If it still happens, check which Wi-Fi band the Mac is on:
  hold **Option** and click the Wi-Fi icon in the menu bar. **2.4 GHz** (channels
  1-14) is the bad one; join the 5 GHz version of the network if there is one
  (often the same name with "5G"), or sit closer to the router. Copy diagnostics
  shows the channel too.
* **Static, or "Getting in sync" over and over, especially on AirPods.** Update to 1.1.5
  or newer. macOS reports Bluetooth headphones' timing with errors that slide by
  several percent; older versions followed those reports and kept jumping to catch
  up. SyncAudio now times playback by the audio the headphones actually take.
* **Plays fine for a minute or two, then breaks up and drops.** Update to 1.1.1 or
  newer. Older versions let a listener fall further and further behind when the Wi-Fi
  couldn't keep up, and macOS slowed the app down once its window was hidden.
* **Still stuck?** Open **Settings → Copy diagnostics** and paste the result to
  whoever is helping you. That's a short summary; the full report (versions, network
  details, audio devices and the last few minutes of sync measurements) is saved as
  **SyncAudio diagnostics.txt** on your Desktop.
* **No sound on the sharing Mac after a crash.** Open SyncAudio again and it puts the
  sound back, or pick your speakers in **Control Center → Sound**.

## How it works

The sharing Mac captures its audio through BlackHole and stamps every 10 ms
block with the exact moment it should come out of the speakers, compressed with
Opus (about 100 kbit/s instead of 1.5 Mbit/s). Every Mac,
including the sharing one, holds each block until that moment. The Macs
match clocks by pinging each other constantly and trusting only the fastest
round trips. Each Mac also corrects for its own speaker's delay as reported
by macOS. Clock drift is fixed by adding or skipping a single sample now and
then, which you can't hear.

In simulated tests with drifting clocks and jittery Wi-Fi, the Macs stay
within about 1 ms of each other. For scale, sound takes about 3 ms to cross
a metre of room.

---

## For developers

The app is `app.py` (a small local web server plus a native window) around
the engine in `syncaudio.py`, with the interface in `ui/index.html`.

**Run from source** (Python 3.9+):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pywebview
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

**Tests** (no audio hardware needed):

```bash
python3 -m unittest discover tests
```

They simulate Macs whose clocks and sound cards drift apart over jittery Wi-Fi,
run real sessions with three Macs over the network with simulated speakers,
and drive the app's API.

### Terminal version

The same engine works from the Terminal, without the app:

```bash
./setup.sh host                 # on the sharing Mac (installs BlackHole via Homebrew)
./setup.sh                      # on the others
./syncaudio host                # share (add --click for the metronome)
./syncaudio join [address]      # listen; finds the host by itself
```

Keys while running: `+`/`-` volume, `[` `]` nudge 1 ms (`{` `}` 10 ms),
`c` cycles both/left/right, `q` quits. `./syncaudio host --delay 250` fixes the
delay instead of choosing it automatically.
