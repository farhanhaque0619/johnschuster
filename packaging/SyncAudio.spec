# PyInstaller recipe for SyncAudio.app (run on a Mac, see .github/workflows/build-mac.yml):
#   SWITCHAUDIO_BIN=/path/to/SwitchAudioSource pyinstaller --noconfirm packaging/SyncAudio.spec
import os

root = os.path.abspath(os.path.join(SPECPATH, ".."))
version = os.environ.get("APP_VERSION", "1.0.0")
switch = os.environ.get("SWITCHAUDIO_BIN")

a = Analysis(
    [os.path.join(root, "app.py")],
    pathex=[root],
    binaries=[(switch, ".")] if switch else [],
    datas=[(os.path.join(root, "ui"), "ui")],
    hiddenimports=["webview.platforms.cocoa"],
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
