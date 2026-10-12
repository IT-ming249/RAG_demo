#!/bin/bash
set -e

# 首次启动时自动导入 dump 文件
if [ -f /init-data/neo4j.dump ] && [ ! -d /data/databases/neo4j ]; then
  echo "=== 正在从 dump 文件导入 Neo4j 数据 ==="
  neo4j-admin database load --from-path=/init-data --overwrite-destination=true neo4j
  echo "=== Neo4j 数据导入完成 ==="
elif [ -d /data/databases/neo4j ]; then
  echo "=== 数据库文件已存在，跳过导入 ==="
fi

# 调用官方 entrypoint 启动 Neo4j
exec /startup/docker-entrypoint.sh neo4j
