#!/usr/bin/env bash
set -euo pipefail
PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$PKG/scripts/run_pair.sh" smoke
