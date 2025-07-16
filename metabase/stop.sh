#!/bin/bash

echo "🛑 Останавливаем Metabase..."

cd "$(dirname "$0")"

docker-compose down

echo "✅ Metabase остановлен"