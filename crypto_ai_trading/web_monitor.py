#!/usr/bin/env python3
"""
Веб-интерфейс для мониторинга обучения
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from flask import Flask, render_template_string, jsonify
import threading

app = Flask(__name__)

# HTML шаблон
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🚀 Crypto AI Training Monitor</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }
        .metric-label {
            color: #888;
            margin-top: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: #2a2a2a;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #444;
        }
        th {
            background-color: #333;
            color: #4CAF50;
        }
        .status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        .status-training { background: #2196F3; }
        .status-completed { background: #4CAF50; }
        .status-failed { background: #f44336; }
        .chart-container {
            margin-top: 30px;
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
        }
        #loss-chart {
            width: 100%;
            height: 400px;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <h1>🚀 Crypto AI Training Monitor</h1>
        
        <div class="metrics" id="metrics">
            <div class="metric-card">
                <div class="metric-value" id="current-epoch">-</div>
                <div class="metric-label">Текущая эпоха</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="train-loss">-</div>
                <div class="metric-label">Train Loss</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="val-loss">-</div>
                <div class="metric-label">Val Loss</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="learning-rate">-</div>
                <div class="metric-label">Learning Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="time-elapsed">-</div>
                <div class="metric-label">Время обучения</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="status">
                    <span class="status status-training">В процессе</span>
                </div>
                <div class="metric-label">Статус</div>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="loss-chart"></canvas>
        </div>

        <h2>📊 История метрик</h2>
        <table id="metrics-table">
            <thead>
                <tr>
                    <th>Эпоха</th>
                    <th>Train Loss</th>
                    <th>Val Loss</th>
                    <th>Train MAE</th>
                    <th>Val MAE</th>
                    <th>Learning Rate</th>
                </tr>
            </thead>
            <tbody id="metrics-tbody">
            </tbody>
        </table>
    </div>

    <script>
        // График loss
        const ctx = document.getElementById('loss-chart').getContext('2d');
        const lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Train Loss',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }, {
                    label: 'Val Loss',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#e0e0e0'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#e0e0e0' },
                        grid: { color: '#444' }
                    },
                    y: {
                        ticks: { color: '#e0e0e0' },
                        grid: { color: '#444' }
                    }
                }
            }
        });

        // Обновление данных
        function updateData() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    // Обновляем метрики
                    document.getElementById('current-epoch').textContent = data.current_epoch || '-';
                    document.getElementById('train-loss').textContent = data.train_loss ? data.train_loss.toFixed(4) : '-';
                    document.getElementById('val-loss').textContent = data.val_loss ? data.val_loss.toFixed(4) : '-';
                    document.getElementById('learning-rate').textContent = data.learning_rate ? data.learning_rate.toExponential(2) : '-';
                    document.getElementById('time-elapsed').textContent = data.time_elapsed || '-';
                    
                    // Обновляем график
                    if (data.history && data.history.length > 0) {
                        lossChart.data.labels = data.history.map(h => h.epoch);
                        lossChart.data.datasets[0].data = data.history.map(h => h.train_loss);
                        lossChart.data.datasets[1].data = data.history.map(h => h.val_loss);
                        lossChart.update();
                        
                        // Обновляем таблицу
                        const tbody = document.getElementById('metrics-tbody');
                        tbody.innerHTML = '';
                        data.history.slice(-10).reverse().forEach(row => {
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td>${row.epoch}</td>
                                <td>${row.train_loss.toFixed(4)}</td>
                                <td>${row.val_loss.toFixed(4)}</td>
                                <td>${row.train_mae ? row.train_mae.toFixed(4) : '-'}</td>
                                <td>${row.val_mae ? row.val_mae.toFixed(4) : '-'}</td>
                                <td>${row.learning_rate.toExponential(2)}</td>
                            `;
                            tbody.appendChild(tr);
                        });
                    }
                });
        }

        // Обновляем каждые 2 секунды
        setInterval(updateData, 2000);
        updateData();
    </script>
</body>
</html>
'''

def find_latest_training_dir():
    """Найти последнюю директорию с обучением"""
    base_dir = Path("experiments/runs")
    if not base_dir.exists():
        return None
    
    training_dirs = sorted(base_dir.glob("training_*"), key=lambda x: x.stat().st_mtime)
    if not training_dirs:
        return None
    
    return training_dirs[-1]

def get_metrics_data():
    """Получить данные метрик"""
    training_dir = find_latest_training_dir()
    if not training_dir:
        return {"error": "No training found"}
    
    # Ищем файл с метриками
    metrics_files = list(training_dir.glob("*_metrics.csv"))
    if not metrics_files:
        return {"error": "No metrics file found"}
    
    metrics_file = metrics_files[0]
    
    try:
        df = pd.read_csv(metrics_file)
        
        # Время обучения
        start_time = datetime.fromtimestamp(training_dir.stat().st_ctime)
        duration = datetime.now() - start_time
        
        # Последняя строка
        last_row = df.iloc[-1] if not df.empty else {}
        
        return {
            "current_epoch": int(last_row.get('epoch', 0)),
            "train_loss": float(last_row.get('train_loss', 0)),
            "val_loss": float(last_row.get('val_loss', 0)),
            "train_mae": float(last_row.get('train_mae', 0)),
            "val_mae": float(last_row.get('val_mae', 0)),
            "learning_rate": float(last_row.get('learning_rate', 0)),
            "time_elapsed": str(duration).split('.')[0],
            "history": df.to_dict('records')
        }
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/metrics')
def metrics():
    return jsonify(get_metrics_data())

if __name__ == '__main__':
    print("🌐 Запуск веб-мониторинга на http://localhost:5000")
    print("📊 TensorBoard доступен на http://localhost:6006")
    app.run(host='0.0.0.0', port=5000, debug=False)