#!/usr/bin/env python3
"""
Универсальный скрипт для применения метаданных из JSON к изображению
Поддерживает как добавление, так и удаление метаданных
"""

import json
import sys
import os
import shutil
from PIL import Image
import piexif
from datetime import datetime

def strip_all_metadata(image_path):
    """
    Полностью удаляет все метаданные из изображения
    """
    img = Image.open(image_path)
    # Создаем новое изображение без метаданных
    data = list(img.getdata())
    image_without_exif = Image.new(img.mode, img.size)
    image_without_exif.putdata(data)
    return image_without_exif

def parse_gps_coordinate(coord_str):
    """
    Парсит GPS координаты из строки типа "8 deg 50' 62.50" N"
    """
    import re
    match = re.match(r'(\d+)\s*deg\s*(\d+)\'\s*([\d.]+)"\s*([NSEW])', coord_str)
    if match:
        deg, min, sec, ref = match.groups()
        return (int(deg), int(min), float(sec)), ref
    return None, None

def parse_altitude(alt_str):
    """Парсит высоту из строки типа "27.3 m Above Sea Level" """
    import re
    match = re.match(r'([\d.]+)\s*m', alt_str)
    if match:
        altitude = float(match.group(1))
        return altitude
    return None

def convert_to_rational(number):
    """Конвертирует число в рациональное представление"""
    if isinstance(number, float):
        return (int(number * 1000), 1000)
    else:
        return (int(number), 1)

def parse_exposure_time(exp_str):
    """Парсит выдержку из строки типа "1/20" """
    if '/' in exp_str:
        num, den = exp_str.split('/')
        return (int(num), int(den))
    return (1, int(exp_str))

def apply_metadata(image_path, json_path, output_path=None, clean_first=True):
    """
    Применяет метаданные из JSON к изображению
    
    Args:
        image_path: путь к исходному изображению
        json_path: путь к JSON файлу с метаданными
        output_path: путь для сохранения (если None, перезаписывает оригинал)
        clean_first: True - сначала удалить все метаданные, False - добавить к существующим
    """
    print("=" * 60)
    print("УНИВЕРСАЛЬНОЕ ПРИМЕНЕНИЕ МЕТАДАННЫХ")
    print("=" * 60)
    
    # Если output_path не указан, работаем с оригиналом
    if output_path is None:
        output_path = image_path
        # Создаем резервную копию
        backup_path = image_path + ".backup"
        print(f"Создание резервной копии: {backup_path}")
        shutil.copy2(image_path, backup_path)
    
    print(f"\nРежим: {'Очистка + применение' if clean_first else 'Добавление к существующим'}")
    
    # Открываем изображение
    if clean_first:
        print("\nШаг 1: Удаление всех существующих метаданных...")
        img = strip_all_metadata(image_path)
        exif_dict = {
            "0th": {},
            "Exif": {},
            "GPS": {},
            "1st": {},
            "thumbnail": None
        }
    else:
        print("\nШаг 1: Чтение существующих метаданных...")
        img = Image.open(image_path)
        try:
            exif_dict = piexif.load(img.info.get('exif', b''))
        except:
            exif_dict = {
                "0th": {},
                "Exif": {},
                "GPS": {},
                "1st": {},
                "thumbnail": None
            }
    
    print("Шаг 2: Чтение метаданных из JSON...")
    
    # Читаем JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Извлекаем метаданные из секции "1"
    metadata = json_data.get("1", {})
    
    print("Шаг 3: Применение метаданных...")
    
    # Счетчики
    applied_fields = 0
    skipped_fields = 0
    
    # Применяем метаданные
    for field_name, field_data in metadata.items():
        if isinstance(field_data, dict):
            value = field_data.get('value', '')
            editable = field_data.get('editable', False)
            
            if not value or value == "(Binary data 25344 bytes)":  # Пропускаем пустые и бинарные
                continue
            
            applied = False
            
            # Основная информация об устройстве
            if field_name == "Make" and editable:
                exif_dict['0th'][piexif.ImageIFD.Make] = value.encode('utf-8')
                applied = True
            
            elif field_name == "Model" and editable:
                exif_dict['0th'][piexif.ImageIFD.Model] = value.encode('utf-8')
                applied = True
            
            elif field_name == "Software" and editable:
                exif_dict['0th'][piexif.ImageIFD.Software] = value.encode('utf-8')
                applied = True
            
            # Даты
            elif field_name == "ModifyDate" and editable:
                exif_dict['0th'][piexif.ImageIFD.DateTime] = value.encode('utf-8')
                applied = True
            
            elif field_name == "DateTimeOriginal" and editable:
                exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = value.encode('utf-8')
                applied = True
            
            elif field_name == "CreateDate" and editable:
                exif_dict['Exif'][piexif.ExifIFD.DateTimeDigitized] = value.encode('utf-8')
                applied = True
            
            # Часовые пояса
            elif field_name == "OffsetTime" and editable:
                exif_dict['Exif'][piexif.ExifIFD.OffsetTime] = value.encode('utf-8')
                applied = True
            
            elif field_name == "OffsetTimeOriginal" and editable:
                exif_dict['Exif'][piexif.ExifIFD.OffsetTimeOriginal] = value.encode('utf-8')
                applied = True
            
            elif field_name == "OffsetTimeDigitized" and editable:
                exif_dict['Exif'][piexif.ExifIFD.OffsetTimeDigitized] = value.encode('utf-8')
                applied = True
            
            # GPS данные
            elif field_name == "GPSLatitude" and editable:
                coords, ref = parse_gps_coordinate(value)
                if coords:
                    deg, min, sec = coords
                    exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = [
                        (deg, 1), (min, 1), (int(sec * 100), 100)
                    ]
                    applied = True
            
            elif field_name == "GPSLatitudeRef" and editable:
                exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = value.encode('utf-8')
                applied = True
            
            elif field_name == "GPSLongitude" and editable:
                coords, ref = parse_gps_coordinate(value)
                if coords:
                    deg, min, sec = coords
                    exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = [
                        (deg, 1), (min, 1), (int(sec * 100), 100)
                    ]
                    applied = True
            
            elif field_name == "GPSLongitudeRef" and editable:
                exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = value.encode('utf-8')
                applied = True
            
            elif field_name == "GPSAltitude" and editable:
                alt = parse_altitude(value)
                if alt:
                    exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = convert_to_rational(alt)
                    exif_dict['GPS'][piexif.GPSIFD.GPSAltitudeRef] = 0  # Above sea level
                    applied = True
            
            elif field_name == "GPSProcessingMethod" and editable:
                exif_dict['GPS'][piexif.GPSIFD.GPSProcessingMethod] = b'ASCII\x00\x00\x00' + value.encode('utf-8')
                applied = True
            
            elif field_name == "GPSTimeStamp" and editable:
                time_parts = value.split(':')
                if len(time_parts) == 3:
                    exif_dict['GPS'][piexif.GPSIFD.GPSTimeStamp] = [
                        (int(time_parts[0]), 1),
                        (int(time_parts[1]), 1),
                        (int(time_parts[2]), 1)
                    ]
                    applied = True
            
            elif field_name == "GPSDateStamp" and editable:
                exif_dict['GPS'][piexif.GPSIFD.GPSDateStamp] = value.encode('utf-8')
                applied = True
            
            elif field_name == "WhiteBalance" and editable:
                wb_value = 0 if value.lower() == "auto" else 1
                exif_dict['Exif'][piexif.ExifIFD.WhiteBalance] = wb_value
                applied = True
            
            elif field_name == "ImageDescription" and editable:
                if value:  # Только если не пустое
                    exif_dict['0th'][piexif.ImageIFD.ImageDescription] = value.encode('utf-8')
                    applied = True
            
            if applied:
                applied_fields += 1
                print(f"  ✓ {field_name}: {value}")
            elif editable:
                skipped_fields += 1
                print(f"  ⚠️ {field_name}: не применено")
    
    print(f"\nПрименено полей: {applied_fields}")
    print(f"Пропущено полей: {skipped_fields}")
    
    # Конвертируем EXIF словарь в байты
    try:
        exif_bytes = piexif.dump(exif_dict)
    except Exception as e:
        print(f"Ошибка при создании EXIF данных: {e}")
        return False
    
    # Сохраняем изображение с новыми EXIF данными
    print(f"\nШаг 4: Сохранение файла...")
    img.save(output_path, exif=exif_bytes, quality=95)
    
    print(f"✅ Файл успешно сохранен: {output_path}")
    
    # Проверяем результат
    print("\nПроверка результата:")
    verify_result(output_path)
    
    return True

def remove_metadata(image_path, output_path=None):
    """
    Полностью удаляет все метаданные из изображения
    """
    print("=" * 60)
    print("УДАЛЕНИЕ ВСЕХ МЕТАДАННЫХ")
    print("=" * 60)
    
    if output_path is None:
        output_path = image_path
        # Создаем резервную копию
        backup_path = image_path + ".backup"
        print(f"Создание резервной копии: {backup_path}")
        shutil.copy2(image_path, backup_path)
    
    print("\nУдаление метаданных...")
    clean_img = strip_all_metadata(image_path)
    
    print("Сохранение очищенного изображения...")
    clean_img.save(output_path, quality=95)
    
    print(f"✅ Метаданные удалены: {output_path}")
    
    # Проверяем результат
    print("\nПроверка результата:")
    try:
        img = Image.open(output_path)
        exif = img.info.get('exif')
        if not exif:
            print("  ✓ Метаданные отсутствуют")
        else:
            print("  ⚠️ Некоторые метаданные могут остаться")
    except Exception as e:
        print(f"  Ошибка при проверке: {e}")
    
    return True

def verify_result(image_path):
    """Проверяет основные метаданные"""
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info.get('exif', b''))
        
        # Проверяем основные поля
        if piexif.ImageIFD.Make in exif_dict['0th']:
            make = exif_dict['0th'][piexif.ImageIFD.Make].decode('utf-8', errors='ignore')
            print(f"  Производитель: {make}")
        
        if piexif.ImageIFD.Model in exif_dict['0th']:
            model = exif_dict['0th'][piexif.ImageIFD.Model].decode('utf-8', errors='ignore')
            print(f"  Модель: {model}")
            
        if piexif.ExifIFD.DateTimeOriginal in exif_dict['Exif']:
            date = exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal].decode('utf-8', errors='ignore')
            print(f"  Дата съемки: {date}")
            
        # GPS
        if piexif.GPSIFD.GPSLatitude in exif_dict['GPS']:
            print("  GPS: ✓ Координаты присутствуют")
                
    except Exception as e:
        print(f"  Метаданные отсутствуют или повреждены")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Управление метаданными изображений')
    parser.add_argument('action', choices=['add', 'remove', 'replace'], 
                        help='Действие: add - добавить к существующим, remove - удалить все, replace - заменить')
    parser.add_argument('-i', '--image', required=True, help='Путь к изображению')
    parser.add_argument('-j', '--json', help='Путь к JSON файлу с метаданными (для add/replace)')
    parser.add_argument('-o', '--output', help='Путь для сохранения результата (по умолчанию перезаписывает оригинал)')
    
    args = parser.parse_args()
    
    # Проверяем существование файлов
    if not os.path.exists(args.image):
        print(f"❌ Изображение не найдено: {args.image}")
        return
    
    if args.action in ['add', 'replace'] and not args.json:
        print("❌ Для действий add/replace требуется указать JSON файл с метаданными")
        return
        
    if args.json and not os.path.exists(args.json):
        print(f"❌ JSON файл не найден: {args.json}")
        return
    
    # Выполняем действие
    if args.action == 'remove':
        success = remove_metadata(args.image, args.output)
    elif args.action == 'add':
        success = apply_metadata(args.image, args.json, args.output, clean_first=False)
    elif args.action == 'replace':
        success = apply_metadata(args.image, args.json, args.output, clean_first=True)
    
    if success:
        print("\n✅ Операция завершена успешно!")
    else:
        print("\n❌ Произошла ошибка!")

if __name__ == "__main__":
    # Если скрипт запущен без аргументов, используем ваши файлы
    if len(sys.argv) == 1:
        print("Использование файлов по умолчанию:")
        json_path = "/Users/ruslan/Downloads/s_193227.json"
        image_path = "/Users/ruslan/Downloads/photo_2025-06-18_17-46-17.jpg"
        
        if os.path.exists(json_path) and os.path.exists(image_path):
            print(f"JSON: {json_path}")
            print(f"Изображение: {image_path}")
            apply_metadata(image_path, json_path, clean_first=True)
        else:
            print("Файлы не найдены. Используйте аргументы командной строки.")
            print("\nПримеры использования:")
            print("  python apply_metadata_universal.py replace -i image.jpg -j metadata.json")
            print("  python apply_metadata_universal.py add -i image.jpg -j metadata.json")
            print("  python apply_metadata_universal.py remove -i image.jpg")
    else:
        main()