#!/bin/bash
# libsacore: the code that runs on CoreAudio's IO threads. Usage: native/build.sh [outdir]
#   macOS: libsacore.dylib (shipped) and libsacore_test.dylib (plus the tests' fake IO thread), for
#          SACORE_ARCHS (default arm64, the app's; setup.sh passes the architecture of its Python)
#   Linux (tests only): libsacore.so; with SACORE_SANITIZE=1 also libsacore_asan.so (ASan + UBSan,
#          float conversions and divisions included: gcc's "undefined" leaves those out)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; out="${1:-$here/..}"
flags="-O2 -std=c11 -Wall -Wextra -Werror -fvisibility=hidden"
case "$(uname -s)" in
  Darwin)
    archs=""; for a in ${SACORE_ARCHS:-arm64}; do archs="$archs -arch $a"; done
    clang $flags -dynamiclib $archs -mmacosx-version-min=14.0 -DSA_NO_FAKE \
      -o "$out/libsacore.dylib" "$here/sacore.c" -framework CoreAudio -framework CoreFoundation
    clang $flags -dynamiclib $archs -mmacosx-version-min=14.0 \
      -o "$out/libsacore_test.dylib" "$here/sacore.c" -framework CoreAudio -framework CoreFoundation ;;
  Linux)
    cc $flags -shared -fPIC -o "$out/libsacore.so" "$here/sacore.c" -lpthread -lm
    if [ "${SACORE_SANITIZE:-0}" = 1 ]; then
      cc -O1 -g -std=c11 -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
        -fno-omit-frame-pointer -shared -fPIC -o "$out/libsacore_asan.so" "$here/sacore.c" -lpthread -lm
    fi ;;
  *) echo "native/build.sh: unsupported system $(uname -s)" >&2; exit 1 ;;
esac
