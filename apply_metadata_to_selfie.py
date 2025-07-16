#!/usr/bin/env python3
"""
Скрипт для применения метаданных из s_193227.json к селфи фото
"""

import json
import sys
from PIL import Image
import piexif
import os
import shutil

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
    Парсит GPS координаты из строки типа "16 deg 41' 41.73" N"
    """
    import re
    match = re.match(r'(\d+)\s*deg\s*(\d+)\'\s*([\d.]+)"\s*([NSEW])', coord_str)
    if match:
        deg, min, sec, ref = match.groups()
        return (int(deg), int(min), float(sec)), ref
    return None, None

def parse_altitude(alt_str):
    """Парсит высоту из строки типа "1494.9 m Above Sea Level" """
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

def apply_metadata_keep_name(image_path, json_path):
    """
    Очищает все метаданные и применяет только те, что в JSON
    Перезаписывает оригинальный файл
    """
    print("=" * 60)
    print("ПРИМЕНЕНИЕ МЕТАДАННЫХ К СЕЛФИ")
    print("=" * 60)
    
    # Создаем резервную копию
    backup_path = image_path + ".backup"
    print(f"Создание резервной копии: {backup_path}")
    shutil.copy2(image_path, backup_path)
    
    print("\nШаг 1: Удаление всех существующих метаданных...")
    
    # Полностью очищаем изображение от метаданных
    clean_img = strip_all_metadata(image_path)
    
    print("Шаг 2: Чтение метаданных из JSON...")
    
    # Читаем JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Извлекаем метаданные из секции "1"
    metadata = json_data.get("1", {})
    
    # Создаем новые EXIF данные с нуля
    exif_dict = {
        "0th": {},
        "Exif": {},
        "GPS": {},
        "1st": {},
        "thumbnail": None
    }
    
    print("Шаг 3: Применение только необходимых метаданных...")
    
    # Счетчик примененных полей
    applied_fields = 0
    
    # Применяем только редактируемые метаданные
    for field_name, field_data in metadata.items():
        if isinstance(field_data, dict) and field_data.get('editable', False):
            value = field_data.get('value', '')
            if not value:  # Пропускаем пустые значения
                continue
                
            applied = False
            
            # Основная информация об устройстве
            if field_name == "Make":
                exif_dict['0th'][piexif.ImageIFD.Make] = value.encode('utf-8')
                applied = True
            
            elif field_name == "Model":
                exif_dict['0th'][piexif.ImageIFD.Model] = value.encode('utf-8')
                applied = True
            
            elif field_name == "Software":
                exif_dict['0th'][piexif.ImageIFD.Software] = value.encode('utf-8')
                applied = True
            
            # Даты
            elif field_name == "ModifyDate":
                exif_dict['0th'][piexif.ImageIFD.DateTime] = value.encode('utf-8')
                applied = True
            
            elif field_name == "DateTimeOriginal":
                exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = value.encode('utf-8')
                applied = True
            
            elif field_name == "CreateDate":
                exif_dict['Exif'][piexif.ExifIFD.DateTimeDigitized] = value.encode('utf-8')
                applied = True
            
            # Часовые пояса
            elif field_name == "OffsetTime":
                exif_dict['Exif'][piexif.ExifIFD.OffsetTime] = value.encode('utf-8')
                applied = True
            
            elif field_name == "OffsetTimeOriginal":
                exif_dict['Exif'][piexif.ExifIFD.OffsetTimeOriginal] = value.encode('utf-8')
                applied = True
            
            elif field_name == "OffsetTimeDigitized":
                exif_dict['Exif'][piexif.ExifIFD.OffsetTimeDigitized] = value.encode('utf-8')
                applied = True
            
            # GPS данные
            elif field_name == "GPSLatitude":
                coords, ref = parse_gps_coordinate(value)
                if coords:
                    deg, min, sec = coords
                    exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = [
                        (deg, 1), (min, 1), (int(sec * 100), 100)
                    ]
                    if ref:
                        exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = ref.encode('utf-8')
                    applied = True
            
            elif field_name == "GPSLatitudeRef":
                exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = value.encode('utf-8')
                applied = True
            
            elif field_name == "GPSLongitude":
                coords, ref = parse_gps_coordinate(value)
                if coords:
                    deg, min, sec = coords
                    exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = [
                        (deg, 1), (min, 1), (int(sec * 100), 100)
                    ]
                    if ref:
                        exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = ref.encode('utf-8')
                    applied = True
            
            elif field_name == "GPSAltitude":
                alt = parse_altitude(value)
                if alt:
                    exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = convert_to_rational(alt)
                    applied = True
            
            elif field_name == "GPSProcessingMethod":
                exif_dict['GPS'][piexif.GPSIFD.GPSProcessingMethod] = b'ASCII\x00\x00\x00' + value.encode('utf-8')
                applied = True
            
            elif field_name == "GPSTimeStamp":
                time_parts = value.split(':')
                if len(time_parts) == 3:
                    exif_dict['GPS'][piexif.GPSIFD.GPSTimeStamp] = [
                        (int(time_parts[0]), 1),
                        (int(time_parts[1]), 1),
                        (int(time_parts[2]), 1)
                    ]
                    applied = True
            
            elif field_name == "GPSDateStamp":
                exif_dict['GPS'][piexif.GPSIFD.GPSDateStamp] = value.encode('utf-8')
                applied = True
            
            elif field_name == "WhiteBalance":
                wb_value = 0 if value.lower() == "auto" else 1
                exif_dict['Exif'][piexif.ExifIFD.WhiteBalance] = wb_value
                applied = True
            
            elif field_name == "ImageDescription":
                if value:  # Только если не пустое
                    exif_dict['0th'][piexif.ImageIFD.ImageDescription] = value.encode('utf-8')
                    applied = True
            
            elif field_name == "RecommendedExposureIndex":
                if int(value) > 0:  # Только если больше 0
                    exif_dict['Exif'][piexif.ExifIFD.RecommendedExposureIndex] = int(value)
                    applied = True
            
            if applied:
                applied_fields += 1
                print(f"  ✓ {field_name}: {value}")
    
    print(f"\nПрименено полей: {applied_fields}")
    
    # Конвертируем EXIF словарь в байты
    try:
        exif_bytes = piexif.dump(exif_dict)
    except Exception as e:
        print(f"Ошибка при создании EXIF данных: {e}")
        return False
    
    # Сохраняем изображение с новыми EXIF данными (перезаписываем оригинал)
    print(f"\nШаг 4: Перезапись оригинального файла...")
    clean_img.save(image_path, exif=exif_bytes, quality=95)
    
    print(f"✅ Файл успешно обновлен: {image_path}")
    
    # Удаляем резервную копию
    try:
        os.remove(backup_path)
        print("✅ Резервная копия удалена")
    except:
        print(f"⚠️  Резервная копия сохранена: {backup_path}")
    
    # Проверяем результат
    print("\nПроверка результата:")
    verify_result(image_path)
    
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
        print(f"Ошибка при чтении EXIF: {e}")

def main():
    # Пути к файлам
    json_path = "/Users/ruslan/Downloads/p2_193545.json"
    image_path = "/Users/ruslan/Downloads/IMG_20250618_193545.jpg"
    
    # Проверяем существование файлов
    if not os.path.exists(json_path):
        print(f"❌ JSON файл не найден: {json_path}")
        return
        
    if not os.path.exists(image_path):
        print(f"❌ Изображение не найдено: {image_path}")
        return
    
    print(f"JSON файл: {json_path}")
    print(f"Изображение: {image_path}")
    
    # Применяем метаданные
    success = apply_metadata_keep_name(image_path, json_path)
    
    if success:
        print("\n✅ Процесс завершен успешно!")
        print(f"Метаданные применены к файлу: {image_path}")
    else:
        print("\n❌ Произошла ошибка!")

if __name__ == "__main__":
    main()