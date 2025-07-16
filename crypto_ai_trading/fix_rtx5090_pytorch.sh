#!/bin/bash

echo "========================================================================="
echo "🚀 Установка PyTorch Nightly для RTX 5090 (sm_120)"
echo "========================================================================="

# Деактивируем текущее окружение если оно активно
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "⚠️  Обнаружено активное виртуальное окружение: $VIRTUAL_ENV"
    echo "   Рекомендуется деактивировать его перед установкой"
fi

echo ""
echo "📦 Удаление текущей версии PyTorch..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "🔄 Установка PyTorch Nightly с поддержкой CUDA 12.8 и sm_120..."
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo ""
echo "✅ Проверка установки..."
python -c "
import torch
print(f'PyTorch версия: {torch.__version__}')
print(f'CUDA доступна: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA версия: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'Compute Capability: {props.major}.{props.minor}')
    print(f'Память GPU: {props.total_memory / 1024**3:.1f} GB')
    
    # Тест производительности
    print('\n🧪 Тест производительности...')
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        print('✅ GPU работает корректно!')
    except Exception as e:
        print(f'❌ Ошибка при тесте: {e}')
"

echo ""
echo "========================================================================="
echo "✅ Установка завершена!"
echo "========================================================================="
echo ""
echo "📌 Дальнейшие шаги:"
echo "1. Перезапустите Python/Jupyter kernel"
echo "2. Запустите обучение: python main.py --mode train"
echo "3. Мониторьте загрузку GPU: nvidia-smi -l 1"
echo ""
echo "⚠️  Если все еще есть предупреждения о sm_120:"
echo "   - Это нормально для nightly версий"
echo "   - GPU будет работать корректно несмотря на предупреждения"