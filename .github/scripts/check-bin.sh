#!/bin/bash
BIN_PATH="target/release/$(basename $(pwd)).exe"
if [ -f "$BIN_PATH" ]; then
  echo "exists_bin=true" >> $GITHUB_OUTPUT
  echo "bin_path=$BIN_PATH" >> $GITHUB_OUTPUT
else
  echo "exists_bin=false" >> $GITHUB_OUTPUT
fi