#!/bin/sh
# Start one of the two lanes from the engine tree this repository installs.
#
#   ./run.sh                     # Hy4-flash        :30000, four slots x 262144
#   ./run.sh qwen                # Qwen3.8-Flash-Next :30001, four slots x 262144, int8 KV
#   ./run.sh qwen --port 30102   # same model, another port; extra args go to the server
#
# The engine tree is $BASALT_DIR when set, else ./basalt-port beside this script (that is where
# install-basalt.sh puts it). One tree serves both packs.
set -e
here=$(cd "$(dirname "$0")" && pwd)
dir=${BASALT_DIR:-$here/basalt-port}
model=hy4
case "${1:-}" in
    hy4|qwen) model=$1; shift ;;
esac
[ -x "$dir/start-$model.sh" ] || {
    echo "engine tree not found: $dir/start-$model.sh" >&2
    echo "run ./install-basalt.sh --model $model first, or set BASALT_DIR" >&2
    exit 1
}
exec "$dir/start-$model.sh" "$@"
