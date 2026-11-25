#!/bin/sh
# Start one of the two lanes, optionally from a named preset.
#
#   ./run.sh                          # Hy4-flash :30000, four slots x 262144 (hy4-agent)
#   ./run.sh --preset hy4-fp8         # the same lane with fp8 KV
#   ./run.sh --list-presets           # names, models and descriptions
#   ./run.sh qwen                     # Qwen3.8-Flash-Next :30001, four slots, int8 KV
#   ./run.sh qwen --port 30102        # extra args go to the server
#   ./run.sh --preset qwen-2 --port 30102
#
# A preset is presets/<name>.conf: MODEL and ARGS (the flags appended to the start script's
# own) plus optional ENV lines (NAME=VALUE exported before the start). The engine tree is
# $BASALT_DIR when set, else ./basalt-port beside this script (that is where install-basalt.sh
# puts it). One tree serves both packs.
set -e
here=$(cd "$(dirname "$0")" && pwd)
dir=${BASALT_DIR:-$here/basalt-port}

die() { echo "$*" >&2; exit 2; }

list_presets() {
    printf '%-14s %-5s %s\n' PRESET MODEL WHAT
    for f in "$here"/presets/*.conf; do
        [ -f "$f" ] || continue
        name=${f##*/}; name=${name%.conf}
        p_model="" p_desc=""
        while IFS= read -r line; do
            case "$line" in ''|'#'*) continue ;; esac
            case "$line" in
                MODEL=*) p_model=${line#MODEL=} ;;
                DESC=*)  p_desc=${line#DESC=} ;;
            esac
        done < "$f"
        printf '%-14s %-5s %s\n' "$name" "${p_model:-?}" "$p_desc"
    done
}

preset="" model=""
while [ $# -gt 0 ]; do
    case "$1" in
        --preset)
            [ $# -ge 2 ] || die "--preset needs a name (see --list-presets)"
            preset=$2; shift 2 ;;
        --list-presets) list_presets; exit 0 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        hy4|qwen)
            [ -z "$model" ] || die "the model is given twice (pick hy4 or qwen)"
            model=$1; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done

if [ -n "$preset" ] && [ -n "$model" ]; then
    die "--preset names its own model; drop '$model'"
fi

if [ -n "$preset" ]; then
    conf="$here/presets/$preset.conf"
    [ -f "$conf" ] || die "unknown preset: $preset (see --list-presets)"
    MODEL="" ARGS="" PRESET_ENV=""
    while IFS= read -r line; do
        case "$line" in ''|'#'*) continue ;; esac
        key=${line%%=*} val=${line#*=}
        case "$key" in
            MODEL) MODEL=$val ;;
            ARGS) ARGS=$val ;;
            ENV)
                case "$val" in
                    *=*) PRESET_ENV="${PRESET_ENV:+$PRESET_ENV
}$val" ;;
                    *) die "$conf: ENV needs NAME=VALUE: $val" ;;
                esac
                ;;
        esac
    done < "$conf"
    [ -n "$MODEL" ] || die "$conf has no MODEL="
    model=$MODEL
    while IFS= read -r e; do
        [ -n "$e" ] || continue
        case "$e" in
            *=*) export "$e" ;;
            *) die "$conf: bad ENV: $e" ;;
        esac
    done <<EOF
$PRESET_ENV
EOF
fi
model=${model:-hy4}

[ -x "$dir/start-$model.sh" ] || {
    echo "engine tree not found: $dir/start-$model.sh" >&2
    echo "run ./install-basalt.sh --model $model first, or set BASALT_DIR" >&2
    exit 1
}
# ARGS is a flag list by design: unquoted, the shell splits it into arguments.
exec "$dir/start-$model.sh" ${ARGS:-} "$@"
