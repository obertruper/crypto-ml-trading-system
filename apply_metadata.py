#!/usr/bin/env python3
"""
Скрипт для применения метаданных из real117_06.json к фотографии
"""

import json
import sys
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import piexif
from datetime import datetime
import os

def parse_gps_coordinate(coord_str):
    """
    Парсит GPS координаты из строки типа "16 deg 41' 41.73" N"
    Возвращает кортеж ((degrees, minutes, seconds), ref)
    """
    import re
    match = re.match(r'(\d+)\s*deg\s*(\d+)\'\s*([\d.]+)"\s*([NSEW])', coord_str)
    if match:
        deg, min, sec, ref = match.groups()
        return (int(deg), int(min), float(sec)), ref
    return None, None

def decimal_to_dms_rational(decimal_degrees):
    """Конвертирует десятичные градусы в градусы, минуты, секунды в формате rational"""
    degrees = int(decimal_degrees)
    minutes = int((decimal_degrees - degrees) * 60)
    seconds = ((decimal_degrees - degrees) * 60 - minutes) * 60
    
    return (
        (degrees, 1),
        (minutes, 1),
        (int(seconds * 100), 100)
    )

def parse_altitude(alt_str):
    """Парсит высоту из строки типа "1494.9 m Above Sea Level" """
    import re
    match = re.match(r'([\d.]+)\s*m', alt_str)
    if match:
        altitude = float(match.group(1))
        return altitude
    return None

def convert_to_rational(number):
    """Конвертирует число в рациональное представление (numerator, denominator)"""
    # Для простоты используем знаменатель 100 или 1000
    if isinstance(number, float):
        return (int(number * 1000), 1000)
    else:
        return (int(number), 1)

def apply_metadata_to_image(image_path, json_path, output_path=None):
    """
    Применяет метаданные из JSON файла к изображению
    """
    # Читаем JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Извлекаем метаданные из секции "1"
    metadata = json_data.get("1", {})
    
    # Открываем изображение
    img = Image.open(image_path)
    
    # Создаем ЧИСТЫЕ EXIF данные (удаляем все существующие)
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    
    # Применяем метаданные
    for field_name, field_data in metadata.items():
        if isinstance(field_data, dict) and field_data.get('editable', False):
            value = field_data.get('value', '')
            key_formatted = field_data.get('key_formatted', '')
            
            # Обработка различных полей
            if field_name == "Make" and value:
                exif_dict['0th'][piexif.ImageIFD.Make] = value.encode('utf-8')
            
            elif field_name == "Model" and value:
                exif_dict['0th'][piexif.ImageIFD.Model] = value.encode('utf-8')
            
            elif field_name == "Software" and value:
                exif_dict['0th'][piexif.ImageIFD.Software] = value.encode('utf-8')
            
            elif field_name == "ImageDescription" and value:
                exif_dict['0th'][piexif.ImageIFD.ImageDescription] = value.encode('utf-8')
            
            elif field_name == "ModifyDate" and value:
                exif_dict['0th'][piexif.ImageIFD.DateTime] = value.encode('utf-8')
            
            elif field_name == "DateTimeOriginal" and value:
                exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = value.encode('utf-8')
            
            elif field_name == "CreateDate" and value:
                exif_dict['Exif'][piexif.ExifIFD.DateTimeDigitized] = value.encode('utf-8')
            
            elif field_name == "OffsetTime" and value:
                exif_dict['Exif'][piexif.ExifIFD.OffsetTime] = value.encode('utf-8')
            
            elif field_name == "OffsetTimeOriginal" and value:
                exif_dict['Exif'][piexif.ExifIFD.OffsetTimeOriginal] = value.encode('utf-8')
            
            elif field_name == "OffsetTimeDigitized" and value:
                exif_dict['Exif'][piexif.ExifIFD.OffsetTimeDigitized] = value.encode('utf-8')
            
            # GPS данные
            elif field_name == "GPSLatitude" and value:
                coords, ref = parse_gps_coordinate(value)
                if coords:
                    deg, min, sec = coords
                    exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = [
                        (deg, 1), (min, 1), (int(sec * 100), 100)
                    ]
            
            elif field_name == "GPSLatitudeRef" and value:
                exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = value.encode('utf-8')
            
            elif field_name == "GPSLongitude" and value:
                coords, ref = parse_gps_coordinate(value)
                if coords:
                    deg, min, sec = coords
                    exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = [
                        (deg, 1), (min, 1), (int(sec * 100), 100)
                    ]
            
            elif field_name == "GPSLongitudeRef" and value:
                exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = value.encode('utf-8')
            
            elif field_name == "GPSAltitude" and value:
                alt = parse_altitude(value)
                if alt:
                    exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = convert_to_rational(alt)
            
            elif field_name == "GPSProcessingMethod" and value:
                # GPS Processing Method требует специальную кодировку
                exif_dict['GPS'][piexif.GPSIFD.GPSProcessingMethod] = b'ASCII\x00\x00\x00' + value.encode('utf-8')
            
            elif field_name == "GPSTimeStamp" and value:
                # Преобразуем время в формат ((час, 1), (минута, 1), (секунда, 1))
                time_parts = value.split(':')
                if len(time_parts) == 3:
                    exif_dict['GPS'][piexif.GPSIFD.GPSTimeStamp] = [
                        (int(time_parts[0]), 1),
                        (int(time_parts[1]), 1),
                        (int(time_parts[2]), 1)
                    ]
            
            elif field_name == "GPSDateStamp" and value:
                exif_dict['GPS'][piexif.GPSIFD.GPSDateStamp] = value.encode('utf-8')
            
            elif field_name == "WhiteBalance" and value:
                # Преобразуем "Auto" в 0, "Manual" в 1
                wb_value = 0 if value.lower() == "auto" else 1
                exif_dict['Exif'][piexif.ExifIFD.WhiteBalance] = wb_value
    
    # Конвертируем EXIF словарь в байты
    try:
        exif_bytes = piexif.dump(exif_dict)
    except Exception as e:
        print(f"Ошибка при создании EXIF данных: {e}")
        # Попробуем удалить проблемные поля и повторить
        if 'thumbnail' in exif_dict:
            exif_dict['thumbnail'] = None
        exif_bytes = piexif.dump(exif_dict)
    
    # Определяем путь для сохранения
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_with_metadata{ext}"
    
    # Сохраняем изображение с новыми EXIF данными
    img.save(output_path, exif=exif_bytes, quality=95)
    
    print(f"Метаданные успешно применены!")
    print(f"Результат сохранен в: {output_path}")
    
    # Проверяем результат
    print("\nПроверка примененных метаданных:")
    verify_exif(output_path)
    
    return output_path

def verify_exif(image_path):
    """Проверяет и выводит EXIF данные изображения"""
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info.get('exif', b''))
        
        print("\nОсновная информация:")
        if piexif.ImageIFD.Make in exif_dict['0th']:
            print(f"  Производитель: {exif_dict['0th'][piexif.ImageIFD.Make].decode('utf-8', errors='ignore')}")
        if piexif.ImageIFD.Model in exif_dict['0th']:
            print(f"  Модель: {exif_dict['0th'][piexif.ImageIFD.Model].decode('utf-8', errors='ignore')}")
        if piexif.ImageIFD.Software in exif_dict['0th']:
            print(f"  ПО: {exif_dict['0th'][piexif.ImageIFD.Software].decode('utf-8', errors='ignore')}")
        
        print("\nДаты:")
        if piexif.ImageIFD.DateTime in exif_dict['0th']:
            print(f"  Дата изменения: {exif_dict['0th'][piexif.ImageIFD.DateTime].decode('utf-8', errors='ignore')}")
        if piexif.ExifIFD.DateTimeOriginal in exif_dict['Exif']:
            print(f"  Дата съемки: {exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal].decode('utf-8', errors='ignore')}")
        
        print("\nGPS данные:")
        if piexif.GPSIFD.GPSLatitude in exif_dict['GPS']:
            lat = exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]
            lat_ref = exif_dict['GPS'].get(piexif.GPSIFD.GPSLatitudeRef, b'N').decode('utf-8')
            print(f"  Широта: {lat[0][0]}° {lat[1][0]}' {lat[2][0]/lat[2][1]}\" {lat_ref}")
        
        if piexif.GPSIFD.GPSLongitude in exif_dict['GPS']:
            lon = exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]
            lon_ref = exif_dict['GPS'].get(piexif.GPSIFD.GPSLongitudeRef, b'E').decode('utf-8')
            print(f"  Долгота: {lon[0][0]}° {lon[1][0]}' {lon[2][0]/lon[2][1]}\" {lon_ref}")
        
        if piexif.GPSIFD.GPSAltitude in exif_dict['GPS']:
            alt = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
            print(f"  Высота: {alt[0]/alt[1]} м")
            
    except Exception as e:
        print(f"Ошибка при чтении EXIF: {e}")

def main():
    # Пути к файлам
    json_path = "/Users/ruslan/PycharmProjects/LLM TRANSFORM/real117_06.json"
    image_path = "/Users/ruslan/PycharmProjects/LLM TRANSFORM/photo_2025-06-18_09-38-58.jpg"
    
    print(f"Применение метаданных из: {json_path}")
    print(f"К изображению: {image_path}")
    
    # Применяем метаданные
    output_path = apply_metadata_to_image(image_path, json_path)
    
if __name__ == "__main__":
    main()