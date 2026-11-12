#!/bin/sh
# Supply the directory containing .kcia-gui, not .kcia-gui itself.
if [ "$#" -ne 1 ]; then
  echo 'Usage: sh scripts/start_kcia_with_previous_data.sh /path/to/old/working/directory' >&2
  exit 2
fi
export KCIA_DATA_DIR="$1"
exec python kcia.py
