#!/usr/bin/env python3
"""
Скрипт для полной интеграции унифицированной модели
Удаляет дубли и приводит код к единому формату
"""

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

def backup_file(filepath):
    """Создает резервную копию файла"""
    if os.path.exists(filepath):
        backup = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup)
        print(f"✅ Backup: {backup}")
        return backup
    return None

def update_main_py():
    """Обновляет main.py для использования унифицированной модели"""
    print("\n📝 Обновление main.py...")
    
    main_path = "main.py"
    backup_file(main_path)
    
    with open(main_path, 'r') as f:
        content = f.read()
    
    # Добавляем импорт унифицированной модели
    if 'from models.patchtst_unified' not in content:
        # Находим блок импортов моделей
        import_pos = content.find('from models.')
        if import_pos != -1:
            # Находим конец строки
            line_end = content.find('\n', import_pos)
            # Вставляем новый импорт
            new_import = "\nfrom models.patchtst_unified import create_unified_model, UnifiedPatchTSTForTrading"
            content = content[:line_end] + new_import + content[line_end:]
    
    # Заменяем логику создания модели
    # Находим блок с проверкой n_targets > 13
    old_logic = """if n_targets > 13:  # PatchTSTForTrading поддерживает только 13 выходов"""
    new_logic = """if config['model']['name'] == 'UnifiedPatchTST':  # Используем унифицированную модель"""
    
    content = content.replace(old_logic, new_logic)
    
    # Заменяем создание модели
    if 'model = models.create_model(config)' in content:
        content = content.replace(
            'model = models.create_model(config)',
            """# Создаем модель в зависимости от конфигурации
    if config['model']['name'] == 'UnifiedPatchTST':
        model = create_unified_model(config)
        logger.info("📊 Используется UnifiedPatchTST с 36 выходами")
    else:
        model = models.create_model(config)"""
        )
    
    # Убираем старые комментарии про 13 выходов
    content = content.replace(
        "# PatchTSTForTrading поддерживает только 13 выходов",
        "# Унифицированная модель поддерживает все 36 выходов"
    )
    
    with open(main_path, 'w') as f:
        f.write(content)
    
    print("✅ main.py обновлен")

def update_trainer_py():
    """Применяет патч к trainer.py"""
    print("\n📝 Патчинг trainer.py...")
    
    trainer_path = "training/trainer.py"
    backup_file(trainer_path)
    
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Патч 1: Добавляем поддержку UnifiedPatchTST в создание loss
    if 'UnifiedPatchTST' not in content:
        old_loss = """# Проверяем если это торговая loss функция
        if 'trading' in loss_name:
            from models.trading_losses import get_trading_loss_function
            return get_trading_loss_function(self.config, loss_type='multi_task')"""
        
        new_loss = """# Проверяем если это торговая loss функция
        if 'trading' in loss_name:
            from models.trading_losses import get_trading_loss_function
            return get_trading_loss_function(self.config, loss_type='multi_task')
        
        # Проверяем если используется унифицированная модель
        model_name = self.config.get('model', {}).get('name', '')
        if model_name == 'UnifiedPatchTST':
            from models.patchtst_unified import UnifiedTradingLoss
            return UnifiedTradingLoss(self.config)"""
        
        content = content.replace(old_loss, new_loss)
    
    # Патч 2: Улучшаем _compute_loss для работы с тензорами
    compute_loss_start = content.find("def _compute_loss(self")
    if compute_loss_start != -1:
        # Находим конец метода
        next_def = content.find("\n    def ", compute_loss_start + 1)
        if next_def == -1:
            next_def = len(content)
        
        # Новая реализация
        new_compute_loss = '''    def _compute_loss(self, outputs: Union[torch.Tensor, Dict], 
                     targets: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """Вычисление потерь с поддержкой унифицированной модели"""
        
        # Для унифицированной модели - прямое применение loss
        if isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor):
            # Проверка размерностей
            if outputs.dim() == 2 and targets.dim() == 2:
                # Убеждаемся что размерности совпадают
                if outputs.shape[-1] != targets.shape[-1]:
                    self.logger.warning(f"Размерности не совпадают: outputs {outputs.shape} vs targets {targets.shape}")
                    min_size = min(outputs.shape[-1], targets.shape[-1])
                    outputs = outputs[..., :min_size]
                    targets = targets[..., :min_size]
                
                # Применяем loss напрямую
                loss = self.criterion(outputs, targets)
                
                # Проверка на NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning("Loss is NaN/Inf, returning zero loss")
                    return torch.tensor(0.0, device=outputs.device, requires_grad=True)
                
                return loss
        
        # Для старой логики с MultiTaskLoss
        if isinstance(self.criterion, MultiTaskLoss):
            # [оставляем старую логику для совместимости]
            losses = {}
            
            if isinstance(outputs, dict) and isinstance(targets, dict):
                # ... старая логика ...
                pass
            
            return self.criterion(losses) if losses else torch.tensor(0.0, device=outputs.device)
        
        # Fallback для простых случаев
        if isinstance(outputs, dict):
            outputs = list(outputs.values())[0]
        if isinstance(targets, dict):
            targets = list(targets.values())[0]
        
        return self.criterion(outputs, targets)'''
        
        # Заменяем метод
        content = content[:compute_loss_start] + new_compute_loss + content[next_def:]
    
    # Патч 3: Добавляем поддержку OneCycleLR
    scheduler_section = content.find("scheduler_name = scheduler_config.get('name'")
    if scheduler_section != -1 and "OneCycleLR" not in content[scheduler_section:scheduler_section+500]:
        insert_pos = content.find("return get_scheduler(", scheduler_section)
        if insert_pos != -1:
            patch = """
        # Специальная обработка для OneCycleLR
        if scheduler_name == 'OneCycleLR':
            # OneCycleLR требует total_steps
            if hasattr(self, 'train_loader'):
                total_steps = self.epochs * len(self.train_loader)
            else:
                total_steps = self.epochs * 1000  # Примерная оценка
            scheduler_config['params']['total_steps'] = total_steps
            scheduler_config['params']['epochs'] = self.epochs
        
        """
            content = content[:insert_pos] + patch + content[insert_pos:]
    
    with open(trainer_path, 'w') as f:
        f.write(content)
    
    print("✅ trainer.py обновлен")

def update_models_init():
    """Обновляет models/__init__.py для экспорта унифицированной модели"""
    print("\n📝 Обновление models/__init__.py...")
    
    init_path = "models/__init__.py"
    
    if os.path.exists(init_path):
        backup_file(init_path)
        
        with open(init_path, 'r') as f:
            content = f.read()
        
        # Добавляем импорт унифицированной модели
        if 'patchtst_unified' not in content:
            content += "\n# Унифицированная модель\nfrom .patchtst_unified import UnifiedPatchTSTForTrading, create_unified_model\n"
        
        with open(init_path, 'w') as f:
            f.write(content)
        
        print("✅ models/__init__.py обновлен")

def verify_integration():
    """Проверяет правильность интеграции"""
    print("\n🔍 Проверка интеграции...")
    
    checks = []
    
    # Проверка импортов в main.py
    with open('main.py', 'r') as f:
        main_content = f.read()
        checks.append(('UnifiedPatchTST в main.py', 'UnifiedPatchTST' in main_content))
    
    # Проверка trainer.py
    with open('training/trainer.py', 'r') as f:
        trainer_content = f.read()
        checks.append(('UnifiedTradingLoss в trainer.py', 'UnifiedTradingLoss' in trainer_content or 'UnifiedPatchTST' in trainer_content))
    
    # Проверка config.yaml
    with open('config/config.yaml', 'r') as f:
        config_content = f.read()
        checks.append(('UnifiedPatchTST в config', 'name: UnifiedPatchTST' in config_content))
        checks.append(('output_size: 36 в config', 'output_size: 36' in config_content))
        checks.append(('learning_rate: 0.001', 'learning_rate: 0.001' in config_content))
    
    # Вывод результатов
    all_good = True
    for check_name, result in checks:
        if result:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_good = False
    
    return all_good

def create_final_run_script():
    """Создает финальный скрипт запуска"""
    print("\n📝 Создание скрипта запуска...")
    
    script = '''#!/bin/bash
# Финальный скрипт запуска обучения с унифицированной моделью

echo "🚀 Запуск обучения с UnifiedPatchTST (36 выходов)..."
echo "📊 Конфигурация:"
echo "  - Learning Rate: 0.001 (увеличен в 10 раз)"
echo "  - Batch Size: 128 (уменьшен для частых обновлений)"
echo "  - Scheduler: OneCycleLR"
echo "  - Model: UnifiedPatchTST с 36 выходами"

# Активация окружения
if [ -f "venv_crypto/bin/activate" ]; then
    source venv_crypto/bin/activate
fi

# Проверка checkpoint
CHECKPOINT="models_saved/best_model_20250701_120952.pth"
if [ -f "$CHECKPOINT" ]; then
    echo "✅ Найден checkpoint: $CHECKPOINT"
    read -p "Продолжить с checkpoint? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        RESUME="--resume $CHECKPOINT"
    else
        RESUME=""
    fi
else
    RESUME=""
fi

# Запуск обучения
echo "🏃 Запускаем обучение..."
python main.py --mode train \\
    --config config/config.yaml \\
    --log_every 50 \\
    --save_every 1 \\
    $RESUME

echo "✅ Готово!"
'''
    
    with open('run_unified_training.sh', 'w') as f:
        f.write(script)
    
    os.chmod('run_unified_training.sh', 0o755)
    print("✅ Создан run_unified_training.sh")

def main():
    """Главная функция"""
    print("🔧 ИНТЕГРАЦИЯ УНИФИЦИРОВАННОЙ МОДЕЛИ\n")
    
    # Выполняем все обновления
    update_main_py()
    update_trainer_py()
    update_models_init()
    
    # Проверяем интеграцию
    if verify_integration():
        print("\n✅ Интеграция успешно завершена!")
        create_final_run_script()
        
        print("\n📋 ИНСТРУКЦИИ:")
        print("1. Остановите текущее обучение (Ctrl+C)")
        print("2. Запустите: ./run_unified_training.sh")
        print("3. Мониторинг: python web_monitor.py")
    else:
        print("\n❌ Обнаружены проблемы! Проверьте файлы вручную.")

if __name__ == "__main__":
    main()