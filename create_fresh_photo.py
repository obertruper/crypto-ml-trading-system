#!/usr/bin/env python3
"""
Скрипт для создания НОВОГО изображения с метаданными
Создает свежий файл, как будто только что снятый на телефоне
"""

import json
import sys
from PIL import Image
import piexif
import os
from datetime import datetime
import io

def parse_gps_coordinate(coord_str):
    """Парсит GPS координаты из строки типа "16 deg 41' 41.73" N" """
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

def parse_exposure_time(exp_str):
    """Парсит выдержку из строки типа "1/20" """
    if '/' in exp_str:
        parts = exp_str.split('/')
        return (int(parts[0]), int(parts[1]))
    return (1, 1)

def create_fresh_photo(original_path, json_path, output_path=None):
    """
    Создает НОВОЕ изображение с метаданными из JSON
    Полностью пересоздает файл, как будто только что снятый
    """
    print("=" * 60)
    print("СОЗДАНИЕ НОВОГО ФОТО С МЕТАДАННЫМИ")
    print("=" * 60)
    
    print("\nШаг 1: Загрузка оригинального изображения...")
    # Открываем оригинал
    original_img = Image.open(original_path)
    
    # Создаем НОВОЕ изображение из пикселей оригинала
    print("Шаг 2: Создание нового изображения...")
    # Конвертируем в RGB если нужно
    if original_img.mode != 'RGB':
        original_img = original_img.convert('RGB')
    
    # Создаем совершенно новое изображение
    new_img = Image.new('RGB', original_img.size)
    new_img.putdata(list(original_img.getdata()))
    
    print("Шаг 3: Чтение метаданных из JSON...")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    metadata = json_data.get("1", {})
    
    # Создаем НОВЫЕ EXIF данные с нуля
    exif_dict = {
        "0th": {},
        "Exif": {},
        "GPS": {},
        "1st": {},
        "thumbnail": None
    }
    
    print("Шаг 4: Применение метаданных...")
    
    applied_fields = 0
    
    # Обрабатываем ВСЕ поля
    for field_name, field_data in metadata.items():
        if not isinstance(field_data, dict):
            continue
            
        value = field_data.get('value', '')
        if not value and value != 0:
            continue
        
        applied = False
        
        # === ОСНОВНАЯ ИНФОРМАЦИЯ ===
        if field_name == "Make":
            exif_dict['0th'][piexif.ImageIFD.Make] = str(value).encode('utf-8')
            applied = True
            
        elif field_name == "Model":
            exif_dict['0th'][piexif.ImageIFD.Model] = str(value).encode('utf-8')
            applied = True
            
        elif field_name == "Software":
            exif_dict['0th'][piexif.ImageIFD.Software] = str(value).encode('utf-8')
            applied = True
            
        elif field_name == "Orientation":
            orientation_map = {
                "Horizontal (normal)": 1,
                "Rotate 90 CW": 6,
                "Rotate 180": 3,
                "Rotate 270 CW": 8
            }
            orientation = orientation_map.get(value, 1)
            exif_dict['0th'][piexif.ImageIFD.Orientation] = orientation
            applied = True
            
        # === РАЗРЕШЕНИЕ ===
        elif field_name == "XResolution":
            exif_dict['0th'][piexif.ImageIFD.XResolution] = (int(value), 1)
            applied = True
            
        elif field_name == "YResolution":
            exif_dict['0th'][piexif.ImageIFD.YResolution] = (int(value), 1)
            applied = True
            
        elif field_name == "ResolutionUnit":
            unit = 2 if value == "inches" else 3
            exif_dict['0th'][piexif.ImageIFD.ResolutionUnit] = unit
            applied = True
            
        # === ДАТЫ И ВРЕМЯ ===
        elif field_name == "ModifyDate":
            exif_dict['0th'][piexif.ImageIFD.DateTime] = str(value).encode('utf-8')
            applied = True
            
        elif field_name == "DateTimeOriginal":
            exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = str(value).encode('utf-8')
            applied = True
            
        elif field_name == "CreateDate":
            exif_dict['Exif'][piexif.ExifIFD.DateTimeDigitized] = str(value).encode('utf-8')
            applied = True
            
        elif field_name == "OffsetTime":
            exif_dict['Exif'][piexif.ExifIFD.OffsetTime] = str(value).encode('utf-8')
            applied = True
            
        elif field_name == "OffsetTimeOriginal":
            exif_dict['Exif'][piexif.ExifIFD.OffsetTimeOriginal] = str(value).encode('utf-8')
            applied = True
            
        elif field_name == "OffsetTimeDigitized":
            exif_dict['Exif'][piexif.ExifIFD.OffsetTimeDigitized] = str(value).encode('utf-8')
            applied = True
            
        # === ПАРАМЕТРЫ СЪЕМКИ ===
        elif field_name == "ExposureTime":
            exp_rational = parse_exposure_time(str(value))
            exif_dict['Exif'][piexif.ExifIFD.ExposureTime] = exp_rational
            applied = True
            
        elif field_name == "FNumber":
            f_value = float(str(value).replace('f/', '')) if 'f/' in str(value) else float(value)
            exif_dict['Exif'][piexif.ExifIFD.FNumber] = (int(f_value * 10), 10)
            applied = True
            
        elif field_name == "ISO":
            exif_dict['Exif'][piexif.ExifIFD.ISOSpeedRatings] = int(value)
            applied = True
            
        elif field_name == "ExposureProgram":
            prog_map = {
                "Not Defined": 0,
                "Manual": 1,
                "Normal": 2,
                "Aperture priority": 3,
                "Shutter priority": 4,
                "Creative": 5,
                "Action": 6,
                "Portrait": 7,
                "Landscape": 8
            }
            exif_dict['Exif'][piexif.ExifIFD.ExposureProgram] = prog_map.get(value, 0)
            applied = True
            
        elif field_name == "Flash":
            flash_value = 0 if "did not fire" in str(value).lower() else 1
            exif_dict['Exif'][piexif.ExifIFD.Flash] = flash_value
            applied = True
            
        elif field_name == "FocalLength":
            focal = float(str(value).replace(' mm', ''))
            exif_dict['Exif'][piexif.ExifIFD.FocalLength] = (int(focal * 100), 100)
            applied = True
            
        elif field_name == "FocalLengthIn35mmFormat":
            focal35 = int(str(value).replace(' mm', ''))
            exif_dict['Exif'][piexif.ExifIFD.FocalLengthIn35mmFilm] = focal35
            applied = True
            
        elif field_name == "MeteringMode":
            meter_map = {
                "Unknown": 0,
                "Average": 1,
                "Center-weighted average": 2,
                "Spot": 3,
                "Multi-spot": 4,
                "Pattern": 5,
                "Partial": 6
            }
            exif_dict['Exif'][piexif.ExifIFD.MeteringMode] = meter_map.get(value, 2)
            applied = True
            
        elif field_name == "WhiteBalance":
            wb_value = 0 if str(value).lower() == "auto" else 1
            exif_dict['Exif'][piexif.ExifIFD.WhiteBalance] = wb_value
            applied = True
            
        elif field_name == "ExposureMode":
            mode_value = 0 if str(value).lower() == "auto" else 1
            exif_dict['Exif'][piexif.ExifIFD.ExposureMode] = mode_value
            applied = True
            
        elif field_name == "SceneCaptureType":
            scene_map = {
                "Standard": 0,
                "Landscape": 1,
                "Portrait": 2,
                "Night": 3
            }
            exif_dict['Exif'][piexif.ExifIFD.SceneCaptureType] = scene_map.get(value, 0)
            applied = True
            
        elif field_name == "DigitalZoomRatio":
            zoom = float(value) if value else 1.0
            exif_dict['Exif'][piexif.ExifIFD.DigitalZoomRatio] = (int(zoom * 100), 100)
            applied = True
            
        elif field_name == "ExposureCompensation":
            ev = int(value)
            exif_dict['Exif'][piexif.ExifIFD.ExposureBiasValue] = (ev, 1)
            applied = True
            
        elif field_name == "MaxApertureValue":
            apex = float(value)
            exif_dict['Exif'][piexif.ExifIFD.MaxApertureValue] = (int(apex * 100), 100)
            applied = True
            
        elif field_name == "BrightnessValue":
            brightness = float(value)
            exif_dict['Exif'][piexif.ExifIFD.BrightnessValue] = (int(brightness * 100), 100)
            applied = True
            
        elif field_name == "ShutterSpeedValue":
            if '/' in str(value):
                exif_dict['Exif'][piexif.ExifIFD.ShutterSpeedValue] = parse_exposure_time(str(value))
                applied = True
                
        # === РАЗМЕРЫ ИЗОБРАЖЕНИЯ ===
        elif field_name == "ExifImageWidth":
            exif_dict['Exif'][piexif.ExifIFD.PixelXDimension] = int(value)
            applied = True
            
        elif field_name == "ExifImageHeight":
            exif_dict['Exif'][piexif.ExifIFD.PixelYDimension] = int(value)
            applied = True
            
        # === GPS ДАННЫЕ ===
        elif field_name == "GPSLatitude":
            coords, ref = parse_gps_coordinate(str(value))
            if coords:
                deg, min, sec = coords
                exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = [
                    (deg, 1), (min, 1), (int(sec * 100), 100)
                ]
                if ref:
                    exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = ref.encode('utf-8')
                applied = True
                
        elif field_name == "GPSLongitude":
            coords, ref = parse_gps_coordinate(str(value))
            if coords:
                deg, min, sec = coords
                exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = [
                    (deg, 1), (min, 1), (int(sec * 100), 100)
                ]
                if ref:
                    exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = ref.encode('utf-8')
                applied = True
                
        elif field_name == "GPSLatitudeRef":
            exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = str(value).encode('utf-8')
            applied = True
            
        elif field_name == "GPSLongitudeRef":
            exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = str(value).encode('utf-8')
            applied = True
            
        elif field_name == "GPSAltitude":
            alt = parse_altitude(str(value))
            if alt:
                exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = convert_to_rational(alt)
                applied = True
                
        elif field_name == "GPSAltitudeRef":
            ref = 0 if "above" in str(value).lower() else 1
            exif_dict['GPS'][piexif.GPSIFD.GPSAltitudeRef] = ref
            applied = True
            
        elif field_name == "GPSSpeed":
            speed = float(value) if value else 0
            exif_dict['GPS'][piexif.GPSIFD.GPSSpeed] = (int(speed * 100), 100)
            applied = True
            
        elif field_name == "GPSSpeedRef":
            ref_map = {"km/h": "K", "mph": "M", "knots": "N"}
            exif_dict['GPS'][piexif.GPSIFD.GPSSpeedRef] = ref_map.get(value, "K").encode('utf-8')
            applied = True
            
        elif field_name == "GPSProcessingMethod":
            exif_dict['GPS'][piexif.GPSIFD.GPSProcessingMethod] = b'ASCII\x00\x00\x00' + str(value).encode('utf-8')
            applied = True
            
        elif field_name == "GPSTimeStamp":
            time_parts = str(value).split(':')
            if len(time_parts) == 3:
                exif_dict['GPS'][piexif.GPSIFD.GPSTimeStamp] = [
                    (int(time_parts[0]), 1),
                    (int(time_parts[1]), 1),
                    (int(time_parts[2]), 1)
                ]
                applied = True
                
        elif field_name == "GPSDateStamp":
            exif_dict['GPS'][piexif.GPSIFD.GPSDateStamp] = str(value).encode('utf-8')
            applied = True
            
        elif field_name == "GPSVersionID":
            version_parts = str(value).split('.')
            if len(version_parts) == 4:
                exif_dict['GPS'][piexif.GPSIFD.GPSVersionID] = bytes([int(p) for p in version_parts])
                applied = True
                
        # === ВЕРСИИ И ФОРМАТЫ ===
        elif field_name == "ExifVersion":
            version = str(value).replace('.', '')
            exif_dict['Exif'][piexif.ExifIFD.ExifVersion] = version.encode('utf-8')
            applied = True
            
        elif field_name == "FlashpixVersion":
            version = str(value).replace('.', '').ljust(4, '0')
            exif_dict['Exif'][piexif.ExifIFD.FlashpixVersion] = version.encode('utf-8')
            applied = True
            
        elif field_name == "ColorSpace":
            cs_value = 1 if str(value).lower() == "srgb" else 65535
            exif_dict['Exif'][piexif.ExifIFD.ColorSpace] = cs_value
            applied = True
            
        elif field_name == "ComponentsConfiguration":
            exif_dict['Exif'][piexif.ExifIFD.ComponentsConfiguration] = b'\x01\x02\x03\x00'
            applied = True
            
        elif field_name == "YCbCrPositioning":
            pos_value = 2 if "co-sited" in str(value).lower() else 1
            exif_dict['0th'][piexif.ImageIFD.YCbCrPositioning] = pos_value
            applied = True
            
        # === ДОПОЛНИТЕЛЬНЫЕ МЕТАДАННЫЕ ===
        elif field_name == "ImageDescription":
            if value:
                exif_dict['0th'][piexif.ImageIFD.ImageDescription] = str(value).encode('utf-8')
                applied = True
                
        elif field_name == "SubSecTime":
            exif_dict['Exif'][piexif.ExifIFD.SubSecTime] = str(value).encode('utf-8')
            applied = True
            
        elif field_name == "SubSecTimeOriginal":
            exif_dict['Exif'][piexif.ExifIFD.SubSecTimeOriginal] = str(value).encode('utf-8')
            applied = True
            
        elif field_name == "SubSecTimeDigitized":
            exif_dict['Exif'][piexif.ExifIFD.SubSecTimeDigitized] = str(value).encode('utf-8')
            applied = True
            
        elif field_name == "LightSource":
            light_map = {
                "Unknown": 0,
                "Daylight": 1,
                "Fluorescent": 2,
                "Tungsten": 3,
                "Flash": 4,
                "Other": 255
            }
            exif_dict['Exif'][piexif.ExifIFD.LightSource] = light_map.get(value, 255)
            applied = True
            
        elif field_name == "SensitivityType":
            sens_map = {
                "Unknown": 0,
                "Standard Output Sensitivity": 1,
                "Recommended Exposure Index": 2,
                "ISO Speed": 3
            }
            exif_dict['Exif'][piexif.ExifIFD.SensitivityType] = sens_map.get(value, 0)
            applied = True
            
        elif field_name == "RecommendedExposureIndex":
            if int(value) > 0:
                exif_dict['Exif'][piexif.ExifIFD.RecommendedExposureIndex] = int(value)
                applied = True
        
        if applied:
            applied_fields += 1
            if applied_fields % 10 == 0:
                print(f"  Применено {applied_fields} полей...")
    
    print(f"\nВсего применено полей: {applied_fields}")
    
    # Создаем миниатюру (thumbnail)
    print("\nШаг 5: Создание миниатюры...")
    thumbnail = new_img.copy()
    thumbnail.thumbnail((160, 120), Image.Resampling.LANCZOS)
    
    # Сохраняем миниатюру в байты
    thumbnail_io = io.BytesIO()
    thumbnail.save(thumbnail_io, format='JPEG', quality=75)
    exif_dict['1st'] = {piexif.ImageIFD.Compression: 6,
                        piexif.ImageIFD.XResolution: (72, 1),
                        piexif.ImageIFD.YResolution: (72, 1),
                        piexif.ImageIFD.ResolutionUnit: 2}
    exif_dict['thumbnail'] = thumbnail_io.getvalue()
    
    # Конвертируем EXIF словарь в байты
    print("Шаг 6: Формирование EXIF данных...")
    try:
        exif_bytes = piexif.dump(exif_dict)
    except Exception as e:
        print(f"Ошибка при создании EXIF данных: {e}")
        # Убираем миниатюру и пробуем снова
        exif_dict['thumbnail'] = None
        exif_dict['1st'] = {}
        exif_bytes = piexif.dump(exif_dict)
    
    # Определяем путь для сохранения
    if output_path is None:
        # Заменяем оригинальный файл
        output_path = original_path
    
    # Сохраняем НОВОЕ изображение
    print(f"\nШаг 7: Сохранение нового изображения...")
    
    # Сохраняем в буфер сначала (чтобы создать совсем новый файл)
    buffer = io.BytesIO()
    new_img.save(buffer, format='JPEG', exif=exif_bytes, quality=95, optimize=True)
    
    # Удаляем старый файл если он существует
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"  Удален старый файл: {output_path}")
    
    # Создаем новый файл из буфера
    with open(output_path, 'wb') as f:
        f.write(buffer.getvalue())
    
    # Устанавливаем время создания и модификации файла в соответствии с EXIF
    if piexif.ExifIFD.DateTimeOriginal in exif_dict['Exif']:
        date_str = exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal].decode('utf-8')
        # Парсим дату из EXIF
        dt = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
        timestamp = dt.timestamp()
        # Устанавливаем время модификации и доступа
        os.utime(output_path, (timestamp, timestamp))
    
    print(f"✅ Файл полностью пересоздан: {output_path}")
    
    # Проверяем результат
    print("\nПроверка результата:")
    verify_fresh_photo(output_path)
    
    return output_path

def verify_fresh_photo(image_path):
    """Проверяет метаданные нового фото"""
    try:
        # Проверяем размер файла
        file_size = os.path.getsize(image_path)
        print(f"  Размер файла: {file_size / 1024 / 1024:.2f} MB")
        
        # Проверяем дату модификации файла
        mod_time = datetime.fromtimestamp(os.path.getmtime(image_path))
        print(f"  Дата файла: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Проверяем EXIF
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info.get('exif', b''))
        
        if piexif.ImageIFD.Make in exif_dict['0th']:
            make = exif_dict['0th'][piexif.ImageIFD.Make].decode('utf-8', errors='ignore')
            print(f"  Производитель: {make}")
        
        if piexif.ImageIFD.Model in exif_dict['0th']:
            model = exif_dict['0th'][piexif.ImageIFD.Model].decode('utf-8', errors='ignore')
            print(f"  Модель: {model}")
        
        if piexif.ExifIFD.DateTimeOriginal in exif_dict['Exif']:
            date = exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal].decode('utf-8', errors='ignore')
            print(f"  Дата съемки: {date}")
        
        if piexif.ExifIFD.ExposureTime in exif_dict['Exif']:
            exp = exif_dict['Exif'][piexif.ExifIFD.ExposureTime]
            print(f"  Выдержка: {exp[0]}/{exp[1]}")
        
        if piexif.ExifIFD.FNumber in exif_dict['Exif']:
            f = exif_dict['Exif'][piexif.ExifIFD.FNumber]
            print(f"  Диафрагма: f/{f[0]/f[1]}")
        
        if piexif.ExifIFD.ISOSpeedRatings in exif_dict['Exif']:
            iso = exif_dict['Exif'][piexif.ExifIFD.ISOSpeedRatings]
            print(f"  ISO: {iso}")
        
        if exif_dict.get('thumbnail'):
            print(f"  Миниатюра: ✓ Присутствует")
        
        if piexif.GPSIFD.GPSLatitude in exif_dict['GPS']:
            print("  GPS: ✓ Полные координаты")
            
    except Exception as e:
        print(f"Ошибка при проверке: {e}")

def main():
    if len(sys.argv) >= 3:
        json_path = sys.argv[1]
        image_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) >= 4 else None
    else:
        print("Использование: python create_fresh_photo.py <json_path> <image_path> [output_path]")
        return
    
    # Проверяем существование файлов
    if not os.path.exists(json_path):
        print(f"❌ JSON файл не найден: {json_path}")
        return
        
    if not os.path.exists(image_path):
        print(f"❌ Изображение не найдено: {image_path}")
        return
    
    print(f"JSON файл: {json_path}")
    print(f"Оригинал: {image_path}")
    
    # Создаем новое фото
    output = create_fresh_photo(image_path, json_path, output_path)
    
    if output:
        print("\n✅ Успешно создано новое фото!")
        print(f"Файл выглядит как только что снятый на {json_path.split('/')[-1].split('.')[0]}")
    else:
        print("\n❌ Произошла ошибка!")

if __name__ == "__main__":
    main()