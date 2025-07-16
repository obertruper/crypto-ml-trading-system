#!/usr/bin/env python3
"""
Скрипт для полной очистки метаданных и применения только нужных из JSON
"""

import json
import sys
from PIL import Image
import piexif
import os

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

def apply_clean_metadata(image_path, json_path, output_path=None):
    """
    Очищает все метаданные и применяет только те, что в JSON
    """
    print("Шаг 1: Удаление всех существующих метаданных...")
    
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
        return None
    
    # Определяем путь для сохранения
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_clean_metadata{ext}"
    
    # Сохраняем изображение с новыми EXIF данными
    clean_img.save(output_path, exif=exif_bytes, quality=95)
    
    print(f"\nШаг 4: Сохранение результата")
    print(f"Файл сохранен: {output_path}")
    
    # Проверяем результат
    print("\nШаг 5: Проверка результата")
    verify_clean_exif(output_path)
    
    return output_path

def verify_clean_exif(image_path):
    """Проверяет, что в файле только нужные метаданные"""
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info.get('exif', b''))
        
        print("\nПримененные метаданные:")
        
        # Проверяем 0th IFD
        if exif_dict['0th']:
            for tag, value in exif_dict['0th'].items():
                tag_name = piexif.TAGS['0th'].get(tag, {}).get('name', f'Unknown({tag})')
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                print(f"  {tag_name}: {value}")
        
        # Проверяем Exif IFD
        if exif_dict['Exif']:
            for tag, value in exif_dict['Exif'].items():
                tag_name = piexif.TAGS['Exif'].get(tag, {}).get('name', f'Unknown({tag})')
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                print(f"  {tag_name}: {value}")
        
        # Проверяем GPS IFD
        if exif_dict['GPS']:
            print("\nGPS данные:")
            for tag, value in exif_dict['GPS'].items():
                tag_name = piexif.TAGS['GPS'].get(tag, {}).get('name', f'Unknown({tag})')
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                print(f"  {tag_name}: {value}")
                
    except Exception as e:
        print(f"Ошибка при чтении EXIF: {e}")

def main():
    # Пути к файлам
    json_path = "/Users/ruslan/PycharmProjects/LLM TRANSFORM/real117_06.json"
    image_path = "/Users/ruslan/PycharmProjects/LLM TRANSFORM/photo_2025-06-18_09-38-58.jpg"
    
    print("=" * 60)
    print("ОЧИСТКА И ПРИМЕНЕНИЕ МЕТАДАННЫХ")
    print("=" * 60)
    print(f"JSON файл: {json_path}")
    print(f"Изображение: {image_path}")
    print("=" * 60)
    
    # Применяем метаданные
    output_path = apply_clean_metadata(image_path, json_path)
    
    if output_path:
        print("\n✅ Процесс завершен успешно!")
    else:
        print("\n❌ Произошла ошибка!")

if __name__ == "__main__":
    main()