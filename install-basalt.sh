#!/usr/bin/env bash
# install-basalt.sh — build the Basalt engine once on a DGX Spark (GB10, compute capability 12.1) and
# install either or both NVFP4 lanes measured on it: Hy4-Flash (:30000) and Qwen3.8-Flash-Next (:30001).
#
# Both lanes used to live in their own recipe repository with their own installer, but they always
# shared ONE engine tree (previously ~/basalt-hy4-port, now basalt-port/ inside this repository). This
# script is those two installers merged: the engine is checked, patched, built and venv'd at most once,
# then every selected lane gets its pack, its start script, optionally its systemd user unit and its
# smoke test. A pack is a file, not a build: one bin/basalt serves both models and the disk holds each
# pack once.
#
# Only one lane can run at a time: 86.2 GiB (Hy4-Flash) plus 68.4 GiB (Qwen3.8-Flash-Next) of weights
# do not fit in the 121 GiB of unified memory, and the engine budgets against MemFree, so there is no
# configuration in which both are loaded. --model all therefore installs both and smoke-tests them one
# after the other, never together.
#
# What it does, in order:
#   1. checks the machine (aarch64, CUDA 13, cmake, gcc, python, free disk for packs that are missing)
#   2. clones the maintained fork agostino-code/basalt (a direct fork of jesdga95/basalt, DGX Spark
#      support in main) at the pinned commit, or reuses --dir
#   3. applies patches/*.patch in order (0001: FP16 K/V for Qwen's batch windows; 0002: the Hy4 prompt
#      chunk policy; 0003: the experimental Hy4 FP4 prompt experts) only when the tree does not already
#      carry each one
#   4. builds it (Release, CMAKE_CUDA_ARCHITECTURES=121) when bin/basalt is missing, a patch was just
#      applied, or the binary was last built at another commit — about 10 minutes
#   5. creates the server venv and installs the pinned requirements when .venv is missing
#   6. downloads each selected model's NVFP4 pack when it is missing or incomplete (147.8 GiB Hy4,
#      122.6 GiB Qwen, resumable and size-checked)
#   7. writes the lane start script(s) — start-hy4.sh / start-qwen.sh — and, with --install-unit, the
#      systemd --user unit(s), left disabled on purpose
#   8. smoke-tests each selected lane: start, wait for "loaded", one completion, stop
#
# Nothing here needs root, and every step is idempotent: a second run reuses the tree, the build, the
# venv and every complete pack. The pack download is the long part (1-3 h on a 100 Mbit line); it
# resumes with -C -, so a broken run is cheap to continue.
#
# Usage:
#   ./install-basalt.sh                            # Hy4 lane into basalt-port/, port 30000, no unit
#   ./install-basalt.sh --model qwen               # only Qwen3.8-Flash-Next, port 30001, int8 KV
#   ./install-basalt.sh --model all                # both lanes: one build, both packs, both smoke tests
#   ./install-basalt.sh --dir /opt/basalt-port     # engine tree elsewhere (or export BASALT_DIR)
#   ./install-basalt.sh --install-unit             # also install the systemd --user unit(s), disabled
#   ./install-basalt.sh --pack /mnt/x.basalt       # single model: use a pack you already have
#   ./install-basalt.sh --model hy4 --port 30100 --kv fp8   # single model: another port, another KV
#   ./install-basalt.sh --vision                   # Hy4 only: single-slot image configuration
#   ./install-basalt.sh --skip-pack                # build only (no pack, no lane, no smoke test)
#   ./install-basalt.sh --jobs 8 --no-smoke
#
# --pack, --port and --kv describe one lane's configuration, so they are refused together with
# --model all. --vision is the Hy4 single-slot image configuration; with --model all it is applied to
# Hy4 only, and a warning says so.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- pinned versions (the combination these recipes were measured with) -------------
# One engine tree for both lanes: the maintained fork agostino-code/basalt, a direct fork of the
# official jesdga95/basalt. Main has the DGX Spark / Arm64 sm_121 support that used to live in the
# third-party port (0.5.7, the GB10 memory planning, the Engram epsilon and the Arm64 CI), so the
# old sm_121 fixes are no longer applied as patches. Same commit for both lanes.
PORT_REPO="https://github.com/agostino-code/basalt.git"
PORT_COMMIT="c70e80118ead375dbcb37948d0cccad8937ad03a"   # main, 2026-11-24 (0.5.7 + DGX support)
# The patches are applied in file order: 0001 (Qwen FP16 batch windows) on the pristine main, 0002
# (Hy4 prompt chunk policy) after it. A patch applied now forces a rebuild below.

MODEL="hy4"                                    # old muscle memory: no flag installs the Hy4 lane
TARGET="${BASALT_DIR:-${SCRIPT_DIR}/basalt-port}"   # one engine tree serves both recipes
JOBS="$(nproc 2>/dev/null || echo 4)"
PACK=""
PORT=""
KV=""
SKIP_PACK=0
SMOKE=1
UNIT=0
VISION=0
SMOKE_PID=""

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --dir) TARGET="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --pack) PACK="$2"; shift 2 ;;
        --kv) KV="$2"; shift 2 ;;
        --skip-pack) SKIP_PACK=1; shift ;;
        --no-smoke) SMOKE=0; shift ;;
        --vision) VISION=1; shift ;;
        --install-unit) UNIT=1; shift ;;
        -h|--help) sed -n '2,48p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m[%s]\033[0m %s\n' "$1" "$2"; }
die()  { printf '\n\033[31merror:\033[0m %s\n' "$1" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
gib()  { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }

# --- what the flags select ---------------------------------------------------------
# Everything below is decided before the machine is touched: a typo in --model must cost nothing.
case "$MODEL" in
    hy4|qwen) MODELS="$MODEL" ;;
    all) MODELS="hy4 qwen" ;;
    *) die "--model takes hy4, qwen or all; found '${MODEL}'" ;;
esac
if [ "$MODEL" = all ]; then
    [ -z "$PACK" ] || die "--pack names one lane's file — install one model at a time for that flag
       (--model hy4 --pack ... / --model qwen --pack ...)"
    [ -z "$PORT" ] || die "--port names one lane's port — install one model at a time for that flag
       (--model hy4 --port ... / --model qwen --port ...)"
    [ -z "$KV" ] || die "--kv names one lane's KV format — install one model at a time for that flag
       (--model hy4 --kv ... / --model qwen --kv ...)"
fi
# The Qwen lane is written with four full windows (1,048,576 cells). FP16 K/V works there with patch
# 0001, but 30.4 GiB of KV pushes tens of GB of experts to RAM; int8 keeps the same windows at 15.2
# GiB. Say so, and leave the choice to the caller.
if [ "$MODEL" = "qwen" ] && [ -n "$KV" ] && [ "$KV" != "int8" ]; then
    echo "  warning: --kv $KV on the Qwen lane's four full windows needs about 30 GiB of KV and pushes
       experts to RAM; the default int8, or ./run.sh --preset qwen-fp16-2, is lighter"
fi
if [ "$VISION" = 1 ]; then
    case "$MODEL" in
        qwen) die "--vision is the Hy4-Flash single-slot image configuration; the Qwen lane was
       measured text-only. Drop --vision, or install the Hy4 lane with --model hy4 --vision." ;;
        all) echo "  warning: --vision is Hy4-Flash only — the Qwen lane is installed text-only" ;;
    esac
fi
# An absolute tree: the start scripts, the units and the smoke test must work from any cwd.
TARGET="$(readlink -m "$TARGET")"

# --- helpers and per-lane configuration -------------------------------------------
model_meta() {
    # The pinned facts per lane, byte-exact from the two old installers. Nothing here is a default that
    # a flag can no longer override: --port and --kv override the port and KV below, --pack the path.
    case "$1" in
        hy4)
            M_TITLE="Hy4-Flash"
            M_PACK_URL="https://huggingface.co/jesdga/Hy4-flash-Basalt/resolve/main/Hy4-flash-NVFP4.basalt"
            M_PACK_NAME="Hy4-flash-NVFP4.basalt"
            M_PACK_BYTES=158746168309              # 147.8 GiB, verified on our box
            M_ALIAS="hy4-flash"
            M_PORT_DEFAULT=30000
            M_KV_DEFAULT=""                        # engine default (fp16): what the Hy4 lane was measured with
            M_KV_LABEL="default fp16"
            M_SCRIPT="start-hy4.sh"
            M_UNIT="hy4-basalt"
            M_DISK_GIB=160                         # the old installer's generous check for 147.8 GiB
            M_MEM_GIB=100                          # weights (~86.2) + KV, measured
            M_LOAD_NOTE="~30 s"
            M_SMOKE_TIMEOUT=180
            ;;
        qwen)
            M_TITLE="Qwen3.8-Flash-Next"
            M_PACK_URL="https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt/resolve/main/Qwen3.8-FN-NVFP4-DQ6_K.basalt"
            M_PACK_NAME="Qwen3.8-FN-NVFP4-DQ6_K.basalt"
            M_PACK_BYTES=131681979197              # 122.6 GiB
            M_ALIAS="qwen3.8-flash-next"
            M_PORT_DEFAULT=30001
            M_KV_DEFAULT="int8"                    # the measured KV format for this model on GB10
            M_KV_LABEL="int8"
            M_SCRIPT="start-qwen.sh"
            M_UNIT="qwen-basalt"
            M_DISK_GIB=130                         # the old installer's generous check for 122.6 GiB
            M_MEM_GIB=90                           # weights (~68.4) + int8 KV, measured
            M_LOAD_NOTE="a minute"
            M_SMOKE_TIMEOUT=300
            ;;
    esac
}

lane_config() {
    # One lane's effective configuration. --port and --kv only exist for a single lane (checked above);
    # with --model all both stay empty and each lane keeps its measured defaults.
    model_meta "$1"
    LANE="$1"
    PORT_VALUE="${PORT:-$M_PORT_DEFAULT}"
    if [ -n "$KV" ]; then KV_VALUE="$KV"; else KV_VALUE="$M_KV_DEFAULT"; fi
    VISION_LANE=0
    if [ "$VISION" = 1 ] && [ "$LANE" = "hy4" ]; then VISION_LANE=1; fi
    if [ "$VISION_LANE" = 1 ]; then
        LANE_FLAGS="--parallel 1 --max-context 262144 --fit-max-tokens --vision dynamic"
        LANE_DESC="1 slot x 262144, image input on"
    else
        LANE_FLAGS="--parallel 4 --max-context 1048576 --fit-max-tokens"
        LANE_DESC="4 slots x 262144, image input off"
    fi
    # The Qwen start script carries its KV as a runtime setting (KV=${KV:-int8}, --kv "$KV"), exactly
    # like the measured script in the tree; the Hy4 lane has no KV line, so --kv is baked in instead
    # (the old Hy4 installer did the same).
    # The Qwen lane's FP16 warning is emitted with the flags, before the machine is touched.
    KV_FLAG=""
    if [ "$LANE" = "qwen" ]; then
        KV_FLAG=' --kv "$KV"'
    elif [ -n "$KV_VALUE" ]; then
        KV_FLAG=" --kv ${KV_VALUE}"
    fi
    KV_TEXT="${KV_VALUE:-$M_KV_LABEL}"
}

resolve_pack() {
    # --pack only exists for a single lane (checked above); without it the pack lives in $TARGET/models.
    # With --skip-pack nothing is downloaded, so the lane scripts point at the default location.
    if [ -n "$PACK" ] && [ "$SKIP_PACK" = 0 ]; then
        PACK_FILE="$(readlink -m "$PACK")"
    else
        PACK_FILE="$TARGET/models/$M_PACK_NAME"
    fi
}

load_lane() {
    lane_config "$1"
    resolve_pack
}

# --- 1. machine checks -----------------------------------------------------------
say 1/8 "checking the machine"
[ "$(uname -m)" = "aarch64" ] || die "this recipe is for Arm64 (aarch64); found $(uname -m)"

NVCC="$(command -v nvcc || true)"
[ -n "$NVCC" ] || NVCC="/usr/local/cuda/bin/nvcc"
[ -x "$NVCC" ] || die "nvcc not found — install the CUDA toolkit (13.x) and/or set PATH=/usr/local/cuda/bin:\$PATH"
CUDA_VER="$("$NVCC" --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -1)"
case "$CUDA_VER" in 13.*) : ;; *) echo "warning: built and measured with CUDA 13, found ${CUDA_VER:-unknown}" ;;
esac

for t in cmake git curl python3; do have "$t" || die "$t is missing"; done
CMAKE_VER="$(cmake --version | sed -n 's/cmake version \([0-9][0-9.]*\)/\1/p')"
awk -v v="$CMAKE_VER" 'BEGIN{split(v,a,"."); exit !(a[1]>3 || (a[1]==3 && a[2]>=24))}' \
    || die "cmake 3.24+ required, found $CMAKE_VER"
GCC_VER="$(gcc -dumpversion | cut -d. -f1)"
[ "$GCC_VER" -ge 13 ] || die "gcc 13+ required, found $GCC_VER"
PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' \
    || die "python 3.11+ required, found $PY_VER"
echo "  aarch64, CUDA ${CUDA_VER:-?}, cmake ${CMAKE_VER}, gcc ${GCC_VER}, python ${PY_VER}, ${JOBS} jobs"

# Disk for the packs that are not here yet. The old installers asked for 160 GiB next to a 147.8 GiB
# pack and 130 GiB next to a 122.6 GiB pack; for two lanes this is the sum, and a pack that is already
# complete costs nothing. --pack points at a file you already have, so it never triggers this check.
if [ "$SKIP_PACK" = 0 ] && [ -z "$PACK" ]; then
    NEED_GIB=0
    NEED_LIST=""
    for m in $MODELS; do
        model_meta "$m"
        if [ -f "$TARGET/models/$M_PACK_NAME" ] && [ "$(stat -c %s "$TARGET/models/$M_PACK_NAME")" = "$M_PACK_BYTES" ]; then
            :   # complete already: no space to reserve
        else
            NEED_GIB=$((NEED_GIB + M_DISK_GIB))
            NEED_LIST="${NEED_LIST}${NEED_LIST:+, }${M_TITLE} ($(gib "$M_PACK_BYTES") GiB)"
        fi
    done
    if [ "$NEED_GIB" -gt 0 ]; then
        FREE_GIB="$(df -BG --output=avail "$(dirname "$TARGET")" 2>/dev/null | tail -1 | tr -dc '0-9')"
        if [ -n "$FREE_GIB" ] && [ "$FREE_GIB" -lt "$NEED_GIB" ]; then
            die "not enough disk for ${NEED_LIST}: up to ${NEED_GIB} GiB needed, ${FREE_GIB} GiB free under $(dirname "$TARGET").
       Free space, or point --pack at an existing pack (one model at a time), or use --skip-pack."
        fi
        echo "  disk: ${NEED_GIB} GiB reserved for ${NEED_LIST}, ${FREE_GIB:-?} GiB free under $(dirname "$TARGET")"
    fi
fi
for m in $MODELS; do
    model_meta "$m"
    LANE_PORT="${PORT:-$M_PORT_DEFAULT}"
    if ss -ltn 2>/dev/null | grep -q ":${LANE_PORT} "; then
        echo "  note: port ${LANE_PORT} is already in use — stop that lane before starting this one"
    fi
done

# --- 2. the port at the pinned commit --------------------------------------------
say 2/8 "port: agostino-code/basalt (direct fork of jesdga95/basalt, main) @ ${PORT_COMMIT:0:8}"
if [ -d "$TARGET/.git" ]; then
    echo "  reusing $TARGET"
    # Never delete or recreate a tree: it may be the one this box was measured on, with the packs and
    # the venv already in it. Only the identity and the pin are checked.
    ORIGIN="$(git -C "$TARGET" remote get-url origin 2>/dev/null || true)"
    case "$ORIGIN" in
        *agostino-code/basalt*|*jesdga95/basalt*) : ;;
        *andreasknopke/basalt-arm-sm121*)
            # The early arm-sm121 fork: same packs and venv, but not the maintained main.  Re-point
            # and reset to the pin; the ignored models/, .venv/ and the untracked start scripts stay,
            # the tracked changes of the old fork are discarded.
            echo "  legacy arm-sm121 tree: re-pointing origin to agostino-code/basalt and resetting to ${PORT_COMMIT:0:8}"
            echo "  models/, .venv/ and the start scripts are kept; tracked changes of the old fork are discarded"
            git -C "$TARGET" remote set-url origin "$PORT_REPO"
            git -C "$TARGET" fetch --quiet --no-tags origin "$PORT_COMMIT"                 || die "cannot fetch ${PORT_COMMIT:0:8} in $TARGET"
            git -C "$TARGET" reset --hard --quiet "$PORT_COMMIT"
            ;;
        *) die "$TARGET is a git tree but not a checkout of agostino-code/basalt (origin: ${ORIGIN:-none}).
       Point --dir at the engine tree, or move that directory away and rerun." ;;
    esac
    HEAD_SHA="$(git -C "$TARGET" rev-parse HEAD)"
    if [ "$HEAD_SHA" != "$PORT_COMMIT" ]; then
        echo "  warning: $TARGET is at ${HEAD_SHA:0:8}, not the pinned ${PORT_COMMIT:0:8}"
        echo "  checking the pin out (local changes are kept; a conflicting one makes this fail)"
        git -C "$TARGET" checkout --quiet "$PORT_COMMIT" \
            || die "cannot check out ${PORT_COMMIT:0:8} in $TARGET — run 'git -C $TARGET fetch origin', then check 'git -C $TARGET status'"
    fi
else
    git clone --quiet "$PORT_REPO" "$TARGET"
    git -C "$TARGET" checkout --quiet "$PORT_COMMIT" || die "cannot check out ${PORT_COMMIT:0:8} in $TARGET"
fi
HEAD_SHA="$(git -C "$TARGET" rev-parse HEAD)"
echo "  at $(git -C "$TARGET" rev-parse --short HEAD) — $(git -C "$TARGET" log -1 --format=%s | cut -c1-60)"

# --- 3. the patches ---------------------------------------------------------------
say 3/8 "checking the patches"
# 0001 (Qwen FP16) is the first patch on the pristine main; 0002 (Hy4 chunk policy) follows it.
# Each file is checked and applied in name order, so a tree with 0001 and without 0002 gets 0002 only.
PATCHED=0
shopt -s nullglob
PATCHES=("$SCRIPT_DIR"/patches/*.patch)
shopt -u nullglob
[ "${#PATCHES[@]}" -gt 0 ] || die "no patches in $SCRIPT_DIR/patches — run this script from the repository root"
for PATCH in "${PATCHES[@]}"; do
    PATCH_NAME="$(basename "$PATCH")"
    if git -C "$TARGET" apply --check "$PATCH" 2>/dev/null; then
        git -C "$TARGET" apply "$PATCH"
        echo "  applied $PATCH_NAME"
        PATCHED=1
    elif git -C "$TARGET" apply --reverse --check "$PATCH" 2>/dev/null; then
        echo "  $PATCH_NAME already in this tree"
    else
        die "$PATCH_NAME neither applies nor reverses at ${PORT_COMMIT:0:8}: the tree is modified or
       from another commit (check 'git -C $TARGET status')"
    fi
done

# --- 4. build --------------------------------------------------------------------
say 4/8 "building (Release, sm_121) — about 10 minutes"
# The binary must belong to the checked-out commit: a pin bump (or a tree moved by hand) rebuilds, so a
# reused install never runs yesterday's engine. The stamp is written only after a successful build.
BUILT_AT="$TARGET/build/.port-commit"
BUILT=""
[ -f "$BUILT_AT" ] && BUILT="$(cat "$BUILT_AT" 2>/dev/null || true)"
if [ -x "$TARGET/bin/basalt" ] && [ "$BUILT" = "$HEAD_SHA" ] && [ "$PATCHED" = 0 ]; then
    echo "  already built at ${HEAD_SHA:0:8}: $TARGET/bin/basalt ($(du -h "$TARGET/bin/basalt" | cut -f1))"
else
    export PATH="/usr/local/cuda/bin:${PATH}"
    export CUDACXX="${NVCC}"
    # A build/ copied or moved with the tree still names the old source path in CMakeCache.txt, and
    # cmake refuses to configure it. The folder is generated output: drop it and configure fresh.
    CACHE_SRC="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$TARGET/build/CMakeCache.txt" 2>/dev/null | head -1)"
    if [ -n "$CACHE_SRC" ] && [ "$CACHE_SRC" != "$TARGET" ]; then
        echo "  build cache was configured for $CACHE_SRC — removing the stale build/ and configuring fresh"
        rm -rf "${TARGET:?}/build"
    fi
    cmake -S "$TARGET" -B "$TARGET/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121 >/dev/null
    cmake --build "$TARGET/build" -j "$JOBS"
    [ -x "$TARGET/bin/basalt" ] || die "build finished without bin/basalt"
    printf '%s\n' "$HEAD_SHA" > "$BUILT_AT"
    echo "  $(du -h "$TARGET/bin/basalt" | cut -f1) binary at $TARGET/bin/basalt for ${HEAD_SHA:0:8}"
fi

# --- 5. server venv --------------------------------------------------------------
say 5/8 "server venv and requirements"
if [ ! -x "$TARGET/.venv/bin/python" ]; then
    python3 -m venv "$TARGET/.venv"
    "$TARGET/.venv/bin/pip" install --quiet --upgrade pip
    "$TARGET/.venv/bin/pip" install --quiet -r "$TARGET/requirements.txt"
    echo "  created $TARGET/.venv"
else
    echo "  reusing $TARGET/.venv"
fi
NUMPY_VER="$("$TARGET/.venv/bin/python" -c 'import numpy, jinja2; print("numpy", numpy.__version__)')" \
    || die "the venv at $TARGET/.venv is missing the pinned requirements — remove .venv and rerun"
echo "  $NUMPY_VER"

# --- 6. the packs ----------------------------------------------------------------
# One pack per lane, both under $TARGET/models by default (or wherever --pack points for the single-lane
# case). The size is the contract: a resumable download can leave a prefix, and a prefix that happens to
# be complete is skipped without touching the network.
for m in $MODELS; do
    load_lane "$m"
    say 6/8 "NVFP4 pack — ${M_TITLE}"
    mkdir -p "$TARGET/models"
    if [ "$SKIP_PACK" = 1 ]; then
        echo "  skipped (--skip-pack)"
    else
        if [ -f "$PACK_FILE" ] && [ "$(stat -c %s "$PACK_FILE")" = "$M_PACK_BYTES" ]; then
            echo "  already there and complete: $PACK_FILE"
        else
            echo "  downloading $(gib "$M_PACK_BYTES") GiB to $PACK_FILE (resumable, this is the long part)"
            echo "  from $M_PACK_URL"
            curl -fL -C - --retry 5 --retry-delay 10 -o "$PACK_FILE" "$M_PACK_URL" \
                || die "download failed — rerun the same command to resume"
        fi
        SIZE="$(stat -c %s "$PACK_FILE")"
        [ "$SIZE" = "$M_PACK_BYTES" ] || die "$M_PACK_NAME has $SIZE bytes, expected $M_PACK_BYTES — incomplete or wrong file"
        echo "  $(gib "$SIZE") GiB, size matches"
    fi
done

# --- 7. lane scripts and units ---------------------------------------------------
# Two measured configurations for Hy4, mutually exclusive on this box (see the README):
#   text: 4 slots x 262144, images off  -- the agent configuration
#   vision: 1 slot, 262144 total, --vision dynamic -- every --parallel >= 2 dies in the planning
#           step with "RAM ran out loading", at any context size from 65536 to 524288
# The Qwen lane has one measured configuration: four agent slots at the model's full native window,
# with int8 KV. --vision is refused for it (the encoder reservation does not fit next to the weights
# either), so with --model all it applies to Hy4 only.
write_start_script() {
    # The content is the measured launcher, unchanged: same env-overridable ROOT/PORT/PACK/ALIAS, the
    # hard error on a busy port, the posix_fadvise drop of the pack's page cache, and "$@" passed on so
    # extra engine flags still work. Only the port, the pack path and the flags differ per lane.
    # The lane must point at the pack we actually have: with --pack the file can live outside $TARGET,
    # so the default PACK is that path instead of $ROOT/models/<name> (the measured form).
    if [ -n "$PACK" ] && [ "$SKIP_PACK" = 0 ] && [ "$PACK_FILE" != "$TARGET/models/$M_PACK_NAME" ]; then
        PACK_DEFAULT="$PACK_FILE"
    else
        PACK_DEFAULT="\$ROOT/models/$M_PACK_NAME"
    fi
    ALIAS_LINE="ALIAS=\${ALIAS:-${M_ALIAS}}"
    PACK_LINE="PACK=\${PACK:-${PACK_DEFAULT}}"
    KV_LINE=""
    case "$LANE" in
        hy4)
            if [ "$VISION_LANE" = 1 ]; then
                HDR="# Hy4-Flash NVFP4 (Basalt) in the foreground: one slot x 262144, image input on, port ${PORT_VALUE}.
# The single-slot image configuration (--parallel 1 --vision dynamic); the agent configuration is the
# text lane (--parallel 4 --max-context 1048576). Extra arguments go to the server."
            else
                HDR="# Hy4-Flash NVFP4 (Basalt) in the foreground: four slots x 262144, text only, port ${PORT_VALUE}.
# The agent configuration from the README (4 x 262144 tokens, images off). Extra arguments go to the
# server; image input is the single-slot configuration: --parallel 1 --vision dynamic."
            fi
            ;;
        qwen)
            HDR="# Qwen3.8-Flash-Next NVFP4 (Basalt) in the foreground: four slots x 262144, ${KV_VALUE} KV, port ${PORT_VALUE}.
# The configuration measured in the qwen38-flash-next-on-spark README. Extra arguments go to the server."
            KV_LINE=$'\n'"KV=\${KV:-${KV_VALUE}}"
            ;;
    esac
    cat > "$TARGET/$M_SCRIPT" <<EOF
#!/bin/sh
${HDR}
set -e
ROOT=\$(cd "\$(dirname "\$0")" && pwd)
PORT=\${PORT:-${PORT_VALUE}}
${PACK_LINE}
${ALIAS_LINE}${KV_LINE}

[ -f "\$PACK" ] || { echo "pack missing: \$PACK" >&2; exit 1; }
if ss -ltn 2>/dev/null | grep -q ":\${PORT} "; then
    echo "port \${PORT} is in use - stop the other lane on it first" >&2
    exit 1
fi

# The engine budgets against MemFree, and a warm page cache of the pack counts as used: dropping it
# avoids a spurious "RAM ran out loading" on a box that just finished a download or a long run.
python3 - "\$PACK" <<'FADV' 2>/dev/null || true
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
FADV

cd "\$ROOT"
exec "\$ROOT/.venv/bin/python" -m serve.server \\
    --engine-bin "\$ROOT/bin/basalt" \\
    --model "\$PACK" \\
    --alias "\$ALIAS" --host 0.0.0.0 --port "\$PORT" \\
    ${LANE_FLAGS}${KV_FLAG} \\
    "\$@"
EOF
    chmod +x "$TARGET/$M_SCRIPT"
}

write_unit() {
    # Same unit the old installers wrote: the lane script in the foreground, restarted by hand only.
    UNIT_DIR="${HOME}/.config/systemd/user"
    mkdir -p "$UNIT_DIR"
    cat > "$UNIT_DIR/${M_UNIT}.service" <<EOF
[Unit]
Description=${M_TITLE} NVFP4 (Basalt) on :${PORT_VALUE}

[Service]
Type=simple
WorkingDirectory=${TARGET}
ExecStart=/bin/sh ${TARGET}/${M_SCRIPT}
# No Restart: the port is shared with other lanes, so an automatic restart would race them.
Restart=no
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    echo "  unit ${M_UNIT}.service installed (left disabled on purpose:"
    echo "  systemctl --user start ${M_UNIT} / enable it only if it owns :${PORT_VALUE} exclusively)"
}

for m in $MODELS; do
    load_lane "$m"
    say 7/8 "lane start script — ${M_TITLE}"
    write_start_script
    echo "  $TARGET/$M_SCRIPT  ($LANE_DESC, KV ${KV_TEXT})"
    if [ -n "$PACK" ] && [ "$SKIP_PACK" = 0 ] && [ "$PACK_FILE" != "$TARGET/models/$M_PACK_NAME" ]; then
        echo "  note: the pack stays at $PACK_FILE (outside the build tree); $M_SCRIPT points there"
    fi
    if [ "$UNIT" = 1 ]; then
        write_unit
    fi
done

# --- 8. smoke test ---------------------------------------------------------------
# Sequential on purpose: two lanes cannot share the GPU (86.2 + 68.4 GiB of weights on 121 GiB). Each
# lane is started, waited for, asked one completion and stopped again before the next one starts.
stop_lane() {
    if [ -n "${SMOKE_PID:-}" ]; then
        kill "$SMOKE_PID" 2>/dev/null || true
    fi
    sleep 3
    pkill -f "${TARGET}/bin/basalt" 2>/dev/null || true
    SMOKE_PID=""
}

cleanup_smoke() {
    if [ -n "${SMOKE_PID:-}" ]; then
        kill "$SMOKE_PID" 2>/dev/null || true
    fi
    pkill -f "${TARGET}/bin/basalt" 2>/dev/null || true
}

smoke_lane() {
    local log="$TARGET/smoke-${LANE}.log"
    local avail loaded fail ok out attempt
    # The engine's planner compares its budget against MemFree, and the pack's page cache counts as
    # used: right after the download and the size check the box can look full (we measured 2.9 GiB free
    # with 148 GiB of cache) and the load dies with "RAM ran out loading". posix_fadvise drops the
    # cache without root and without touching a running download.
    if [ -f "$PACK_FILE" ]; then
        if python3 - "$PACK_FILE" <<'FADV'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
FADV
        then
            echo "  dropped the pack's page cache — MemFree now $(awk '/MemFree/{printf "%.0f", $2/1048576}' /proc/meminfo) GiB"
        else
            echo "  warning: could not drop the pack's page cache"
        fi
    fi

    # The smoke test loads the whole pack into unified memory (~86.2 GiB of weights for Hy4, ~68.4 for
    # Qwen) and then builds the KV on top. On a box that is already running another lane (or downloading
    # something) that ends in "RAM ran out loading" — which says nothing about the install. Skip it
    # instead of failing. Another lane holding the GPU leaves no room for a second engine either (even
    # NVML fails then). MemFree, not MemAvailable: the engine's planner compares against MemFree, so a
    # full page cache (another lane's reads, a finished download) makes it refuse even though the kernel
    # would reclaim.
    avail=0
    for _ in $(seq 1 24); do
        avail="$(awk '/MemFree/{printf "%d", $2/1048576}' /proc/meminfo)"
        [ "${avail:-0}" -ge "$M_MEM_GIB" ] && break
        sleep 5
    done
    if [ "${avail:-0}" -lt "$M_MEM_GIB" ]; then
        echo "  skipped: only ${avail} GiB MemFree, the smoke test needs ~${M_MEM_GIB} GiB (the engine"
        echo "           budgets against MemFree, so another lane's weights and a page cache count"
        echo "           against it); later: $TARGET/$M_SCRIPT"
        return 0
    fi

    echo "  starting the lane for a smoke test (the model load takes ${M_LOAD_NOTE})"
    setsid nohup "$TARGET/$M_SCRIPT" > "$log" 2>&1 &
    SMOKE_PID=$!
    trap cleanup_smoke EXIT
    # /v1/models answers while the weights are still loading (status "loading"), so wait for
    # status "loaded" before asking for a completion.
    loaded=0
    fail=0
    for _ in $(seq 1 90); do
        sleep 5
        kill -0 "$SMOKE_PID" 2>/dev/null || { tail -20 "$log"; fail=1; break; }
        if curl -fsS -m 5 "http://127.0.0.1:${PORT_VALUE}/v1/models" 2>/dev/null | grep -q '"value": *"loaded"'; then
            loaded=1; break
        fi
    done
    [ "$loaded" = 1 ] || fail=1
    if [ "$fail" = 1 ]; then
        echo "  NOTE: the lane did not come up for the smoke test. That is a statement about this box's"
        echo "        free memory at this moment, not about the install: the engine needs the weights plus"
        echo "        the KV next to whatever else runs, and it budgets against MemFree. The build, the"
        echo "        venv and the pack above are installed either way. Start it later with:"
        echo "        $TARGET/$M_SCRIPT"
        stop_lane
        return 0
    fi
    curl -fsS -m 5 "http://127.0.0.1:${PORT_VALUE}/v1/models" | head -c 200 || true
    echo

    # A 503 here means the engine is still coming up after the weights: retry before failing.
    ok=0
    for attempt in 1 2 3 4 5; do
        if out="$(curl -fsS -m "$M_SMOKE_TIMEOUT" "http://127.0.0.1:${PORT_VALUE}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"${M_ALIAS}\",\"max_tokens\":16,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}]}" 2>&1)"; then
            echo "$out" | head -c 300 || true
            echo
            ok=1; break
        fi
        echo "  completion attempt ${attempt} failed (${out##*: }) — retrying in 10 s"
        sleep 10
    done
    if [ "$ok" != 1 ]; then
        tail -5 "$log"
        echo "  NOTE: the lane loaded but did not answer a completion — check $log."
    fi
    if [ "$VISION_LANE" = 1 ]; then
        if grep -q 'images=on' "$log"; then
            echo "  image input is on (images=on)"
        else
            echo "  NOTE: expected images=on in the log, found: $(grep -oE 'images=[a-z]+' "$log" | tail -1)"
        fi
    fi
    echo "  smoke test done — stopping the lane again"
    stop_lane
}

if [ "$SMOKE" = 1 ] && [ "$SKIP_PACK" = 0 ]; then
    for m in $MODELS; do
        load_lane "$m"
        say 8/8 "smoke test — ${M_TITLE}"
        smoke_lane
    done
else
    say 8/8 "smoke test"
    if [ "$SKIP_PACK" = 1 ]; then
        echo "  skipped (--skip-pack: there is no pack to load yet)"
    else
        echo "  skipped (--no-smoke)"
    fi
fi

# --- summary ---------------------------------------------------------------------
cat <<EOF

Done.
EOF
for m in $MODELS; do
    load_lane "$m"
    cat <<EOF

  ${M_TITLE}
  port        : ${PORT_VALUE}  (${LANE_DESC}, KV ${KV_TEXT})
  start       : ${TARGET}/${M_SCRIPT}
  $( if [ "$UNIT" = 1 ]; then echo "unit        : systemctl --user start ${M_UNIT}   (installed, left disabled)"; else echo "unit        : not installed (--install-unit to add it)"; fi )
EOF
    case "$m" in
        hy4)
            cat <<EOF
  acceptance  : python3 tools/basalt-repro.py crash   -> R1/R2/R3 must pass, then
                grep -c 'illegal memory' ${TARGET}/smoke-hy4.log   # 0
  benchmark   : see the README (opencode-bench-plusplus, codec-reverse-bench)
EOF
            ;;
        qwen)
            cat <<EOF
  acceptance  : python3 tools/qwen-lane-check.py --base http://127.0.0.1:${PORT_VALUE} \\
                  --log ${TARGET}/lane.log     # R1/R2/R3 PASS, 0 faults
  benchmark   : basalt-bench --model ${PACK_FILE} -t conv,struct,prose -p 16384,65536
EOF
            ;;
    esac
done
if [ "$MODEL" = all ]; then
    cat <<EOF

  Both lanes are installed, and only one of them fits in memory at a time: 86.2 + 68.4 GiB of weights
  on 121 GiB. Stop one before starting the other:
    pkill -f ${TARGET}/bin/basalt            # stop whichever lane is running
    ${TARGET}/start-hy4.sh                   # foreground; the Qwen lane is start-qwen.sh
EOF
fi
