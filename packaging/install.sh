#!/bin/bash
# Installs or updates SyncAudio. In Terminal:
#   curl -fsSL https://github.com/farhanhaque0619/johnschuster/releases/latest/download/install.sh | bash
# (Downloading this way also skips macOS's "can't verify" prompt.)
set -euo pipefail
DMG="${SYNCAUDIO_DMG:-https://github.com/farhanhaque0619/johnschuster/releases/latest/download/SyncAudio.dmg}"
DEST="${SYNCAUDIO_DEST:-/Applications}"

work="$(mktemp -d)"
cleanup() { hdiutil detach -quiet -force "$work/mount" 2>/dev/null || true; rm -rf "$work"; }
trap cleanup EXIT

echo "Downloading SyncAudio…"
if [[ "$DMG" == https://* ]]; then curl -fL --progress-bar -o "$work/SyncAudio.dmg" "$DMG"; else cp "$DMG" "$work/SyncAudio.dmg"; fi
mkdir "$work/mount"
hdiutil attach -nobrowse -readonly -noautoopen -quiet -mountpoint "$work/mount" "$work/SyncAudio.dmg"

if pgrep -x SyncAudio >/dev/null; then
  echo "Quitting SyncAudio…"
  osascript -e 'quit app "SyncAudio"' >/dev/null 2>&1 || true
  for _ in $(seq 1 25); do pgrep -x SyncAudio >/dev/null || break; sleep 0.2; done
  pkill -x SyncAudio 2>/dev/null || true
fi

rm -rf "$DEST/SyncAudio.app"
ditto "$work/mount/SyncAudio.app" "$DEST/SyncAudio.app"
version=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$DEST/SyncAudio.app/Contents/Info.plist")
echo "SyncAudio $version is installed in $DEST."
if [[ -z "${SYNCAUDIO_NO_OPEN:-}" ]]; then open "$DEST/SyncAudio.app"; fi
