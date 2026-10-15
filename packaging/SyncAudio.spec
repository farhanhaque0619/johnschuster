# PyInstaller recipe for SyncAudio.app (run on a Mac, see .github/workflows/build-mac.yml):
#   SACORE_LIB=$PWD/libsacore.dylib SWITCHAUDIO_BIN=/path/to/SwitchAudioSource pyinstaller --noconfirm packaging/SyncAudio.spec
import os

root = os.path.abspath(os.path.join(SPECPATH, ".."))
version = os.environ.get("APP_VERSION", "1.0.0")
switch = os.environ.get("SWITCHAUDIO_BIN")
opus = os.environ.get("OPUS_LIB")      # libopus.0.dylib; the self-test fails without it
sacore = os.environ["SACORE_LIB"]      # libsacore.dylib (native/build.sh): no sound at all without it

a = Analysis(
    [os.path.join(root, "app.py")],
    pathex=[root],
    binaries=[(sacore, ".")] + ([(switch, ".")] if switch else []) + ([(opus, ".")] if opus else []),
    datas=[(os.path.join(root, "ui"), "ui")] + (
        [(os.path.join(root, "build_info.json"), ".")] if os.path.exists(os.path.join(root, "build_info.json")) else []),
    hiddenimports=["webview.platforms.cocoa"],     # (sacore and audio_engine: app.py imports them)
    excludes=["tkinter", "matplotlib", "PIL"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SyncAudio", console=False,
          argv_emulation=False, codesign_identity=None)
coll = COLLECT(exe, a.binaries, a.datas, name="SyncAudio")
app = BUNDLE(
    coll,
    name="SyncAudio.app",
    icon=os.path.join(SPECPATH, "icon.png"),
    bundle_identifier="io.github.farhanhaque0619.syncaudio",
    version=version,
    info_plist={
        "CFBundleDisplayName": "SyncAudio",
        "CFBundleShortVersionString": version,
        "LSMinimumSystemVersion": "14.0",   # numpy wheels need Sonoma
        "LSApplicationCategoryType": "public.app-category.music",
        "NSHighResolutionCapable": True,
        # No App Nap: hidden behind Spotify, a napping app can't keep the audio coming in time.
        "NSAppSleepDisabled": True,
        # BlackHole is an input device, so capturing it needs microphone permission.
        "NSMicrophoneUsageDescription":
            "SyncAudio listens to the BlackHole audio driver to share what this Mac is playing. "
            "It never uses your real microphone.",
        # The window shows the app's own page from http://127.0.0.1.
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
        "NSLocalNetworkUsageDescription":
            "SyncAudio finds and connects to other Macs on your Wi-Fi so they can play music in sync.",
    },
)
