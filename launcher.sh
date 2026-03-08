#!/usr/bin/env bash
# Wrapper script for Llama Cluster Launcher
cd "$(dirname "$0")"
exec python3 llamaherder.py "$@"
