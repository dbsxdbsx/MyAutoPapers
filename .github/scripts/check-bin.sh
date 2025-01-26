#!/bin/bash
# 假设项目名称为my_auto_papers（根据实际Cargo.toml中的name字段调整）
BIN_PATH="target/release/my_auto_papers.exe"
if [ -f "$BIN_PATH" ]; then
  echo "exists_bin=true" >> $GITHUB_OUTPUT
  echo "bin_path=$BIN_PATH" >> $GITHUB_OUTPUT
else
  echo "exists_bin=false" >> $GITHUB_OUTPUT
fi