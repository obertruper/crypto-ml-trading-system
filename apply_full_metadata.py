#!/usr/bin/env python3
"""
Скрипт для применения ВСЕХ метаданных из JSON файла к изображению
Включает как редактируемые, так и нередактируемые поля для полной эмуляции фото с телефона
"""

import json
import sys
from PIL import Image
import piexif
import os
import shutil
from datetime import datetime

def strip_all_metadata(image_path):
    """Полностью удаляет все метаданные из изображения"""
    img = Image.open(image_path)
    data = list(img.getdata())
    image_without_exif = Image.new(img.mode, img.size)
    image_without_exif.putdata(data)
    return image_without_exif

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

def apply_all_metadata(image_path, json_path, output_path=None):
    """
    Применяет ВСЕ метаданные из JSON файла к изображению
    Включая нередактируемые поля для полной эмуляции
    """
    print("=" * 60)
    print("ПРИМЕНЕНИЕ ПОЛНЫХ МЕТАДАННЫХ")
    print("=" * 60)
    
    # Создаем резервную копию
    backup_path = image_path + ".backup"
    print(f"Создание резервной копии: {backup_path}")
    shutil.copy2(image_path, backup_path)
    
    print("\nШаг 1: Удаление всех существующих метаданных...")
    clean_img = strip_all_metadata(image_path)
    
    print("Шаг 2: Чтение метаданных из JSON...")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    metadata = json_data.get("1", {})
    
    # Создаем новые EXIF данные с нуля
    exif_dict = {
        "0th": {},
        "Exif": {},
        "GPS": {},
        "1st": {},
        "thumbnail": None
    }
    
    print("Шаг 3: Применение ВСЕХ метаданных (включая параметры камеры)...")
    
    applied_fields = 0
    
    # Обрабатываем ВСЕ поля, не только редактируемые
    for field_name, field_data in metadata.items():
        if not isinstance(field_data, dict):
            continue
            
        value = field_data.get('value', '')
        if not value and value != 0:  # Пропускаем только действительно пустые
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
            # Преобразуем текст в число
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
            unit = 2 if value == "inches" else 3  # 2 = inches, 3 = cm
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
            # Преобразуем f/1.8 в рациональное число
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
            # Простое преобразование
            flash_value = 0 if "did not fire" in str(value).lower() else 1
            exif_dict['Exif'][piexif.ExifIFD.Flash] = flash_value
            applied = True
            
        elif field_name == "FocalLength":
            # Извлекаем числовое значение из "3.8 mm"
            focal = float(str(value).replace(' mm', ''))
            exif_dict['Exif'][piexif.ExifIFD.FocalLength] = (int(focal * 100), 100)
            applied = True
            
        elif field_name == "FocalLengthIn35mmFormat":
            # Извлекаем числовое значение из "27 mm"
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
            # APEX value in EV
            ev = int(value)
            exif_dict['Exif'][piexif.ExifIFD.ExposureBiasValue] = (ev, 1)
            applied = True
            
        elif field_name == "MaxApertureValue":
            # APEX value
            apex = float(value)
            exif_dict['Exif'][piexif.ExifIFD.MaxApertureValue] = (int(apex * 100), 100)
            applied = True
            
        elif field_name == "BrightnessValue":
            # APEX brightness
            brightness = float(value)
            exif_dict['Exif'][piexif.ExifIFD.BrightnessValue] = (int(brightness * 100), 100)
            applied = True
            
        elif field_name == "ShutterSpeedValue":
            # APEX shutter speed
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
            # Версия GPS IFD
            version_parts = str(value).split('.')
            if len(version_parts) == 4:
                exif_dict['GPS'][piexif.GPSIFD.GPSVersionID] = bytes([int(p) for p in version_parts])
                applied = True
                
        # === ВЕРСИИ И ФОРМАТЫ ===
        elif field_name == "ExifVersion":
            # Версия Exif (обычно 0220 для 2.2)
            version = str(value).replace('.', '')
            exif_dict['Exif'][piexif.ExifIFD.ExifVersion] = version.encode('utf-8')
            applied = True
            
        elif field_name == "FlashpixVersion":
            # Версия Flashpix (обычно 0100 для 1.0)
            version = str(value).replace('.', '').ljust(4, '0')
            exif_dict['Exif'][piexif.ExifIFD.FlashpixVersion] = version.encode('utf-8')
            applied = True
            
        elif field_name == "ColorSpace":
            # 1 = sRGB, 65535 = Uncalibrated
            cs_value = 1 if str(value).lower() == "srgb" else 65535
            exif_dict['Exif'][piexif.ExifIFD.ColorSpace] = cs_value
            applied = True
            
        elif field_name == "ComponentsConfiguration":
            # Обычно 1,2,3,0 для YCbCr
            exif_dict['Exif'][piexif.ExifIFD.ComponentsConfiguration] = b'\x01\x02\x03\x00'
            applied = True
            
        elif field_name == "YCbCrPositioning":
            # 1 = centered, 2 = co-sited
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
            # ISO sensitivity type
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
            print(f"  ✓ {field_name}: {value}")
    
    print(f"\nПрименено полей: {applied_fields}")
    
    # Конвертируем EXIF словарь в байты
    try:
        exif_bytes = piexif.dump(exif_dict)
    except Exception as e:
        print(f"Ошибка при создании EXIF данных: {e}")
        return False
    
    # Определяем путь для сохранения
    if output_path is None:
        output_path = image_path
    
    # Сохраняем изображение с новыми EXIF данными
    print(f"\nШаг 4: Сохранение файла...")
    clean_img.save(output_path, exif=exif_bytes, quality=95)
    
    print(f"✅ Файл успешно обновлен: {output_path}")
    
    # Удаляем резервную копию
    try:
        os.remove(backup_path)
        print("✅ Резервная копия удалена")
    except:
        print(f"⚠️  Резервная копия сохранена: {backup_path}")
    
    # Проверяем результат
    print("\nПроверка результата:")
    verify_full_exif(output_path)
    
    return True

def verify_full_exif(image_path):
    """Проверяет основные метаданные"""
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info.get('exif', b''))
        
        # Основная информация
        if piexif.ImageIFD.Make in exif_dict['0th']:
            make = exif_dict['0th'][piexif.ImageIFD.Make].decode('utf-8', errors='ignore')
            print(f"  Производитель: {make}")
        
        if piexif.ImageIFD.Model in exif_dict['0th']:
            model = exif_dict['0th'][piexif.ImageIFD.Model].decode('utf-8', errors='ignore')
            print(f"  Модель: {model}")
        
        # Параметры съемки
        if piexif.ExifIFD.ExposureTime in exif_dict['Exif']:
            exp = exif_dict['Exif'][piexif.ExifIFD.ExposureTime]
            print(f"  Выдержка: {exp[0]}/{exp[1]}")
        
        if piexif.ExifIFD.FNumber in exif_dict['Exif']:
            f = exif_dict['Exif'][piexif.ExifIFD.FNumber]
            print(f"  Диафрагма: f/{f[0]/f[1]}")
        
        if piexif.ExifIFD.ISOSpeedRatings in exif_dict['Exif']:
            iso = exif_dict['Exif'][piexif.ExifIFD.ISOSpeedRatings]
            print(f"  ISO: {iso}")
        
        if piexif.ExifIFD.DateTimeOriginal in exif_dict['Exif']:
            date = exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal].decode('utf-8', errors='ignore')
            print(f"  Дата съемки: {date}")
        
        # GPS
        if piexif.GPSIFD.GPSLatitude in exif_dict['GPS']:
            print("  GPS: ✓ Полные координаты присутствуют")
            
    except Exception as e:
        print(f"Ошибка при чтении EXIF: {e}")

def main():
    # Получаем аргументы командной строки
    if len(sys.argv) >= 3:
        json_path = sys.argv[1]
        image_path = sys.argv[2]
    else:
        # Значения по умолчанию для тестирования
        json_path = "/Users/ruslan/Downloads/s_193227.json"
        image_path = "/Users/ruslan/Downloads/IMG_20250618_193227.jpg"
    
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
    success = apply_all_metadata(image_path, json_path)
    
    if success:
        print("\n✅ Процесс завершен успешно!")
        print("Фото теперь выглядит как только что снятое на телефоне!")
    else:
        print("\n❌ Произошла ошибка!")

if __name__ == "__main__":
    main()