#!/usr/bin/env bash
# One-time setup.
#   ./setup.sh host    on the Mac that plays the music (also installs BlackHole)
#   ./setup.sh         on your friend's Mac
set -euo pipefail
cd "$(dirname "$0")"

role="${1:-join}"
case "$role" in
  host|join) ;;
  *) echo "usage: ./setup.sh [host]"; exit 1 ;;
esac

if [[ "$(uname)" != "Darwin" ]]; then
  echo "syncaudio is made for macOS."; exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
    if [[ -x "$b" ]]; then eval "$("$b" shellenv)"; break; fi
  done
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is needed: install it from https://brew.sh (one command), then run this again."
  exit 1
fi

py="$(brew --prefix)/bin/python3"
if [[ ! -x "$py" ]]; then
  echo "Installing Python..."
  brew install python
fi

echo "Installing Python packages into .venv ..."
"$py" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

# The small piece of C that feeds the speakers (native/sacore.c) is built here, with Apple's compiler.
xcode-select -p >/dev/null 2>&1 || { echo "Run: xcode-select --install   (Apple's free command line tools), then this again."; exit 1; }
echo "Building SyncAudio's audio core..."
# for the architecture this Python runs as (an Intel Mac, or Homebrew's Intel Python, can't load arm64)
SACORE_ARCHS="$(.venv/bin/python -c 'import platform; print(platform.machine())')" bash native/build.sh

if [[ "$role" == "host" ]]; then
  echo "Installing BlackHole (virtual audio cable) and SwitchAudioSource..."
  brew list switchaudio-osx >/dev/null 2>&1 || brew install switchaudio-osx
  brew list --cask blackhole-2ch >/dev/null 2>&1 || brew install --cask blackhole-2ch
  if ! .venv/bin/python syncaudio.py devices | grep -qi blackhole; then
    echo "Restarting macOS audio so BlackHole shows up (may ask for your password)..."
    sudo killall coreaudiod || true
    sleep 3
  fi
  if .venv/bin/python syncaudio.py devices | grep -qi blackhole; then
    echo; echo "All set. Start sharing with:  ./syncaudio host"
  else
    echo; echo "BlackHole isn't showing up yet - restart the Mac, then run:  ./syncaudio host"
  fi
else
  echo; echo "All set. Join with:  ./syncaudio join"
fi
