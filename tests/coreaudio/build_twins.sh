#!/bin/bash
# Two more loopback devices for the Mac CI, built from BlackHole's own source (v0.7.1, the version of
# the pkg the app installs). BlackHole reports kLatency_Frame_Size as both its safety offset and its
# stream latency (BlackHole.c:2741, 3165; device latency is always 0):
#   "SyncTestBT 2ch"   3840 + 3840 frames = 160 ms of fixed latency, what AirPods report: PortAudio
#                      treats it exactly as it treats AirPods (1.1.6's latency=0.1 -> its smallest buffer)
#   "SyncTestSpk 2ch"  no latency: a second plain loopback, for the two-device tests
# Optional: if this fails, the workflow carries on with stock BlackHole only. Needs Xcode and sudo;
# coreaudiod must be restarted afterwards to load them. Build logs: /tmp/bh-bt.log, /tmp/bh-spk.log.
set -euo pipefail
rm -rf /tmp/bh
git clone --quiet --depth 1 --branch v0.7.1 https://github.com/ExistentialAudio/BlackHole /tmp/bh
cd /tmp/bh
# v0.7.1 #defines kLatency_Frame_Size with no #ifndef around it (later versions have one), so the -D below
# would lose to the source's 0 with only a warning, and the BT twin would quietly have no latency.
perl -0pi -e 's/^#define\s+kLatency_Frame_Size\s+0[ \t]*$/#ifndef kLatency_Frame_Size\n#define kLatency_Frame_Size 0\n#endif/m' \
  BlackHole/BlackHole.c
grep -q '^#ifndef kLatency_Frame_Size' BlackHole/BlackHole.c || { echo "couldn't make kLatency_Frame_Size settable"; exit 1; }

build() {   # $1 driver name, $2 kLatency_Frame_Size, $3 tag
  local name="$1" lat="$2" id="io.github.syncaudio.test.$3" log="/tmp/bh-$3.log"
  rm -rf build
  # Unsigned here: the Info.plist changes below, and it is signed after that.
  if ! xcodebuild -project BlackHole.xcodeproj -configuration Release -target BlackHole \
      CONFIGURATION_BUILD_DIR=build PRODUCT_BUNDLE_IDENTIFIER="$id" CODE_SIGNING_ALLOWED=NO \
      GCC_PREPROCESSOR_DEFINITIONS='$GCC_PREPROCESSOR_DEFINITIONS kNumber_Of_Channels=2 kLatency_Frame_Size='"$lat"' kPlugIn_BundleID=\"'"$id"'\" kDriver_Name=\"'"$name"'\"' \
      > "$log" 2>&1; then
    tail -40 "$log"; echo "building $name failed (whole log: $log)"; return 1
  fi
  local plist=build/BlackHole.driver/Contents/Info.plist old
  # Each driver needs its own CFPlugIn factory UUID, as BlackHole's create_installer.sh does it.
  old=$(plutil -extract CFPlugInFactories xml1 -o - "$plist" | grep -oE '[0-9A-Fa-f]{8}(-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}' | head -1) || true
  [ -n "$old" ] || { echo "no CFPlugIn factory in $plist"; return 1; }
  sed -i '' "s/$old/$(uuidgen)/g" "$plist"
  codesign --force --deep --sign - build/BlackHole.driver
  sudo rm -rf "/Library/Audio/Plug-Ins/HAL/$name.driver"
  sudo cp -R build/BlackHole.driver "/Library/Audio/Plug-Ins/HAL/$name.driver"
  echo "installed \"$name 2ch\" (kLatency_Frame_Size $lat)"
}
build SyncTestBT 3840 bt
build SyncTestSpk 0 spk
