"""
Визуализация результатов обучения и торговли
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path

from utils.logger import get_logger

class TradingVisualizer:
    """Класс для создания графиков и визуализаций"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger("Visualizer")
        
        # Настройки стиля
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Директории для сохранения
        self.plots_dir = Path("results/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_training_progress(self, 
                             metrics_history: Dict, 
                             save_path: Optional[str] = None) -> str:
        """График прогресса обучения модели"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(metrics_history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, metrics_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, metrics_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy/MAE
        if 'train_accuracy' in metrics_history:
            axes[0, 1].plot(epochs, metrics_history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
            axes[0, 1].plot(epochs, metrics_history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
            axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('Accuracy')
        else:
            axes[0, 1].plot(epochs, metrics_history.get('train_mae', []), 'b-', label='Train MAE', linewidth=2)
            axes[0, 1].plot(epochs, metrics_history.get('val_mae', []), 'r-', label='Val MAE', linewidth=2)
            axes[0, 1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('MAE')
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'learning_rate' in metrics_history:
            axes[1, 0].plot(epochs, metrics_history['learning_rate'], 'g-', linewidth=2)
            axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient Norm (если доступно)
        if 'grad_norm' in metrics_history:
            axes[1, 1].plot(epochs, metrics_history['grad_norm'], 'purple', linewidth=2)
            axes[1, 1].set_title('Gradient Norm', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Placeholder для четвертого графика
            axes[1, 1].text(0.5, 0.5, 'Additional metrics\nwill appear here', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, alpha=0.6)
            axes[1, 1].set_title('Additional Metrics', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Сохранение
        if save_path is None:
            save_path = self.plots_dir / "training_progress.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"График прогресса обучения сохранен: {save_path}")
        return str(save_path)
    
    def plot_backtest_results(self, results: Dict, save_path: Optional[str] = None) -> str:
        """Комплексная визуализация результатов бэктестирования"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Equity Curve', 'Drawdown',
                'Monthly Returns', 'Trade Distribution',
                'Performance by Symbol', 'Risk Metrics'
            ),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Подготовка данных
        equity_df = pd.DataFrame(results['equity_curve'])
        trades_df = pd.DataFrame(results['trades_log'])
        
        if not equity_df.empty:
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            
            # 1. Equity Curve
            fig.add_trace(
                go.Scatter(
                    x=equity_df['timestamp'],
                    y=equity_df['equity'],
                    mode='lines',
                    name='Equity',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Активные позиции (вторичная ось)
            fig.add_trace(
                go.Scatter(
                    x=equity_df['timestamp'],
                    y=equity_df['active_positions'],
                    mode='lines',
                    name='Active Positions',
                    line=dict(color='orange', width=1),
                    yaxis='y2'
                ),
                row=1, col=1, secondary_y=True
            )
            
            # 2. Drawdown
            equity_series = equity_df['equity']
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak * 100
            
            fig.add_trace(
                go.Scatter(
                    x=equity_df['timestamp'],
                    y=drawdown,
                    mode='lines',
                    fill='tonexty',
                    name='Drawdown %',
                    line=dict(color='red', width=1),
                    fillcolor='rgba(255,0,0,0.2)'
                ),
                row=1, col=2
            )
        
        # 3. Monthly Returns (если есть достаточно данных)
        if not equity_df.empty and len(equity_df) > 30:
            equity_df['month'] = equity_df['timestamp'].dt.to_period('M')
            monthly_returns = equity_df.groupby('month')['equity'].agg(['first', 'last'])
            monthly_returns['return'] = (monthly_returns['last'] - monthly_returns['first']) / monthly_returns['first'] * 100
            
            colors = ['green' if x > 0 else 'red' for x in monthly_returns['return']]
            
            fig.add_trace(
                go.Bar(
                    x=monthly_returns.index.astype(str),
                    y=monthly_returns['return'],
                    name='Monthly Return %',
                    marker_color=colors
                ),
                row=2, col=1
            )
        
        # 4. Trade Distribution
        if not trades_df.empty:
            completed_trades = trades_df[trades_df['action'] == 'close']
            if not completed_trades.empty:
                fig.add_trace(
                    go.Histogram(
                        x=completed_trades['pnl'],
                        nbinsx=20,
                        name='PnL Distribution',
                        marker_color='skyblue',
                        opacity=0.7
                    ),
                    row=2, col=2
                )
        
        # 5. Performance by Symbol
        if 'performance_by_symbol' in results and results['performance_by_symbol']:
            symbols = list(results['performance_by_symbol'].keys())
            pnls = [results['performance_by_symbol'][s]['total_pnl'] for s in symbols]
            
            colors = ['green' if x > 0 else 'red' for x in pnls]
            
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=pnls,
                    name='PnL by Symbol',
                    marker_color=colors
                ),
                row=3, col=1
            )
        
        # 6. Risk Metrics (Radar Chart)
        metrics = ['Sharpe Ratio', 'Win Rate %', 'Profit Factor', 'Calmar Ratio']
        values = [
            min(results.get('sharpe_ratio', 0), 3),  # Нормализуем Sharpe
            results.get('win_rate_pct', 0),
            min(results.get('profit_factor', 0), 5),  # Нормализуем PF
            min(results.get('calmar_ratio', 0), 3)   # Нормализуем Calmar
        ]
        
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name='Risk Metrics'
            ),
            row=3, col=2
        )
        
        # Обновление layout
        fig.update_layout(
            height=1200,
            title_text=f"Backtest Results - Total Return: {results.get('total_return_pct', 0):.2f}%",
            title_x=0.5,
            showlegend=True
        )
        
        # Сохранение
        if save_path is None:
            save_path = self.plots_dir / "backtest_results.html"
        
        fig.write_html(save_path)
        
        self.logger.info(f"График результатов бэктестирования сохранен: {save_path}")
        return str(save_path)
    
    def plot_feature_importance(self, 
                               feature_importance: Dict, 
                               top_n: int = 20,
                               save_path: Optional[str] = None) -> str:
        """График важности признаков"""
        
        # Сортировка признаков по важности
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importance = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        
        # Горизонтальный bar plot
        bars = plt.barh(range(len(features)), importance, color='skyblue', alpha=0.8)
        
        # Добавление значений на бары
        for i, (feature, imp) in enumerate(sorted_features):
            plt.text(imp + 0.001, i, f'{imp:.3f}', va='center', fontsize=10)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Инвертируем ось Y для отображения самых важных сверху
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        # Сохранение
        if save_path is None:
            save_path = self.plots_dir / "feature_importance.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"График важности признаков сохранен: {save_path}")
        return str(save_path)
    
    def plot_prediction_analysis(self, 
                                actual: np.ndarray, 
                                predicted: np.ndarray,
                                symbol: str = "",
                                save_path: Optional[str] = None) -> str:
        """Анализ качества предсказаний"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Actual vs Predicted Scatter
        axes[0, 0].scatter(actual, predicted, alpha=0.6, s=20)
        axes[0, 0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'Actual vs Predicted {symbol}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Добавляем R²
        r2 = np.corrcoef(actual, predicted)[0, 1]**2
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # 2. Residuals
        residuals = actual - predicted
        axes[0, 1].scatter(predicted, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution of Residuals
        axes[1, 0].hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue')
        axes[1, 0].axvline(residuals.mean(), color='red', linestyle='--', 
                          label=f'Mean: {residuals.mean():.3f}')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Time Series (если есть временная последовательность)
        if len(actual) > 50:  # Показываем только если достаточно данных
            sample_indices = np.linspace(0, len(actual)-1, min(200, len(actual)), dtype=int)
            axes[1, 1].plot(sample_indices, actual[sample_indices], 'b-', label='Actual', alpha=0.7)
            axes[1, 1].plot(sample_indices, predicted[sample_indices], 'r-', label='Predicted', alpha=0.7)
            axes[1, 1].set_xlabel('Time Index')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_title('Time Series Comparison')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Not enough data\nfor time series plot', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        # Сохранение
        if save_path is None:
            save_path = self.plots_dir / f"prediction_analysis_{symbol}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"График анализа предсказаний сохранен: {save_path}")
        return str(save_path)
    
    def plot_correlation_matrix(self, 
                              data: pd.DataFrame, 
                              features: List[str],
                              save_path: Optional[str] = None) -> str:
        """Матрица корреляций признаков"""
        
        # Выбираем только числовые признаки
        numeric_features = [f for f in features if f in data.columns and data[f].dtype in ['int64', 'float64']]
        
        if len(numeric_features) < 2:
            self.logger.warning("Недостаточно числовых признаков для корреляционной матрицы")
            return ""
        
        # Ограничиваем количество признаков для читаемости
        if len(numeric_features) > 50:
            numeric_features = numeric_features[:50]
        
        corr_matrix = data[numeric_features].corr()
        
        plt.figure(figsize=(12, 10))
        
        # Создаем маску для верхнего треугольника
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Сохранение
        if save_path is None:
            save_path = self.plots_dir / "correlation_matrix.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Матрица корреляций сохранена: {save_path}")
        return str(save_path)
    
    def plot_price_predictions(self, 
                             market_data: pd.DataFrame,
                             predictions: Dict,
                             symbol: str,
                             save_path: Optional[str] = None) -> str:
        """График предсказаний цены с сигналами"""
        
        symbol_data = market_data[market_data['symbol'] == symbol].copy()
        
        if symbol_data.empty:
            self.logger.warning(f"Нет данных для символа {symbol}")
            return ""
        
        symbol_data = symbol_data.sort_values('datetime').tail(100)  # Последние 100 точек
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=(f'{symbol} Price and Predictions', 'Volume', 'Signals'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # 1. Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=symbol_data['datetime'],
                open=symbol_data['open'],
                high=symbol_data['high'],
                low=symbol_data['low'],
                close=symbol_data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # 2. Volume
        fig.add_trace(
            go.Bar(
                x=symbol_data['datetime'],
                y=symbol_data['volume'],
                name='Volume',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 3. Signals (фиктивные для демонстрации)
        signal_times = symbol_data['datetime'].iloc[::10]  # Каждая 10-я точка
        signal_values = np.random.choice([1, 0, -1], size=len(signal_times))
        
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in signal_values]
        
        fig.add_trace(
            go.Scatter(
                x=signal_times,
                y=signal_values,
                mode='markers',
                marker=dict(size=10, color=colors),
                name='Signals'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text=f"{symbol} Trading Analysis",
            xaxis_rangeslider_visible=False
        )
        
        # Сохранение
        if save_path is None:
            save_path = self.plots_dir / f"price_predictions_{symbol}.html"
        
        fig.write_html(save_path)
        
        self.logger.info(f"График предсказаний цены сохранен: {save_path}")
        return str(save_path)
    
    def create_dashboard(self, 
                        training_results: Optional[Dict] = None,
                        backtest_results: Optional[Dict] = None,
                        feature_importance: Optional[Dict] = None) -> str:
        """Создание комплексного дашборда"""
        
        dashboard_path = self.plots_dir / "trading_dashboard.html"
        
        # HTML шаблон дашборда
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto AI Trading Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    text-align: center;
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 10px;
                }}
                .section {{
                    background-color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }}
                .metric {{
                    background-color: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #7f8c8d;
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Crypto AI Trading System Dashboard</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Секция результатов бэктестирования
        if backtest_results:
            html_content += f"""
            <div class="section">
                <h2>📈 Backtest Results</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">{backtest_results.get('total_return_pct', 0):.2f}%</div>
                        <div class="metric-label">Total Return</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{backtest_results.get('sharpe_ratio', 0):.2f}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{backtest_results.get('max_drawdown_pct', 0):.2f}%</div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{backtest_results.get('win_rate_pct', 0):.1f}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{backtest_results.get('total_trades', 0)}</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{backtest_results.get('profit_factor', 0):.2f}</div>
                        <div class="metric-label">Profit Factor</div>
                    </div>
                </div>
            </div>
            """
        
        # Секция результатов обучения
        if training_results:
            html_content += f"""
            <div class="section">
                <h2>🧠 Training Results</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">{training_results.get('final_loss', 0):.4f}</div>
                        <div class="metric-label">Final Loss</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{training_results.get('best_val_loss', 0):.4f}</div>
                        <div class="metric-label">Best Val Loss</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{training_results.get('epochs_trained', 0)}</div>
                        <div class="metric-label">Epochs Trained</div>
                    </div>
                </div>
            </div>
            """
        
        # Важность признаков
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features_html = "<ul>"
            for feature, importance in top_features:
                features_html += f"<li><strong>{feature}</strong>: {importance:.4f}</li>"
            features_html += "</ul>"
            
            html_content += f"""
            <div class="section">
                <h2>🔍 Top 10 Important Features</h2>
                {features_html}
            </div>
            """
        
        html_content += """
            <div class="section">
                <h2>📊 Available Reports</h2>
                <ul>
                    <li><a href="training_progress.png">Training Progress</a></li>
                    <li><a href="backtest_results.html">Detailed Backtest Results</a></li>
                    <li><a href="feature_importance.png">Feature Importance</a></li>
                    <li><a href="correlation_matrix.png">Feature Correlation Matrix</a></li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Дашборд создан: {dashboard_path}")
        return str(dashboard_path)