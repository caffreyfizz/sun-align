import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from solar_center import find_otsu_threshold, shift_image
from solar_rotation import pair_angle, ring_profiles, rotate

# ----------------------------- Пути -----------------------------
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "pictures"
CENTERED_DIR = None  # None — не сохранять промежуточные кадры
ALIGNED_DIR = ROOT / "aligned_pictures"
METRICS_DIR = ROOT / "metrics"
FRAMES_CSV = METRICS_DIR / "frames.csv"     # строка на каждый кадр
SUMMARY_CSV = METRICS_DIR / "summary.csv"   # общие метрики серии
TIME_FORMAT = "%Y%m%d_%H%M%S"  

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CENTERED_PREFIX = "cntrd_"
ALIGNED_PREFIX = "algnd_"


# ----------------------- Параметры поворота ----------------------
N_POINTS = 720   # точек на кольце, шаг 0.5
RINGS = list(np.linspace(0.70, 0.95, 12)) + list(np.linspace(1.04, 1.20, 18))

# ------------------------- Тестовый режим ------------------------
# Кадры перед обработкой искусственно поворачиваются на эти углы
# None — рабочий режим на настоящих снимках
TEST_ANGLES = [0, 3.5, -5.89, 83, -2.9, 6.33, -9.4, 4.1, -7, 10.5, -4.75, 30.12]


def list_frames(folder):
    """Пути ко всем картинкам папки, по имени"""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def frame_center(img):
    h, w = img.shape[:2]
    return (w - 1) / 2, (h - 1) / 2


def center_frame(color):
    """Сдвигает Солнце в центр кадра

    Возвращает (centered, info) или None, если кадр брак
    """
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    sun = find_otsu_threshold(gray)
    if sun is None or sun[0] is None:
        return None
    center_x, center_y, threshold, radius = sun

    target_x, target_y = frame_center(color)
    dx = round(target_x - center_x)
    dy = round(target_y - center_y)
    centered = shift_image(color, dx, dy)

    info = {
        "target_x": target_x, "target_y": target_y,
        "center_x": center_x, "center_y": center_y,
        "dx": dx, "dy": dy,
        "after_x": center_x + dx, "after_y": center_y + dy,
        "radius": radius, "threshold": threshold,
    }
    return centered, info


def frame_time(path):
    """Время съёмки из имени файла (20260301_001741_...) или None"""
    try:
        return datetime.strptime(path.stem[:15], TIME_FORMAT)
    except ValueError:
        return None


def save_csv(path, rows):
    """rows — список словарей с одинаковыми ключами"""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rate_per_hour(hours, values):
    """Наклон прямой values(hours): скорость изменения в единицах за час"""
    if len(hours) < 2 or max(hours) == min(hours):
        return float("nan")
    slope, _ = np.polyfit(hours, values, 1)
    return slope


def summarize(records, total, elapsed):
    """Общие метрики серии по строкам frames.csv"""
    def col(key):
        return np.array([r[key] for r in records], dtype=float)

    offset = np.hypot(col("dx"), col("dy"))                       # сдвиг до центра, px
    residual = np.maximum(np.abs(col("after_x") - col("target_x")),
                          np.abs(col("after_y") - col("target_y")))  # остаток после сдвига по худшей оси
    cx, cy, angle = col("center_x"), col("center_y"), col("angle")
    jump = np.hypot(np.diff(cx), np.diff(cy))                     # скачок центра между соседними кадрами
    angle_step = np.abs(np.diff(angle))                           # скачок угла между соседними кадрами
    hours = col("hours")

    rows = [
        ("кадров всего", total, ""),
        ("обработано", len(records), ""),
        ("пропущено", total - len(records), ""),
        ("сдвиг до центра, среднее", offset.mean(), "px"),
        ("сдвиг до центра, максимум", offset.max(), "px"),
        ("остаток после сдвига (по оси), максимум", residual.max(), "px"),
        ("разброс центра по x (max-min)", cx.max() - cx.min(), "px"),
        ("разброс центра по y (max-min)", cy.max() - cy.min(), "px"),
        ("скачок центра между кадрами, среднее", jump.mean() if len(jump) else 0, "px"),
        ("скачок центра между кадрами, максимум", jump.max() if len(jump) else 0, "px"),
        ("дрейф центра по x", rate_per_hour(hours, cx), "px/ч"),
        ("дрейф центра по y", rate_per_hour(hours, cy), "px/ч"),
        ("угол, среднее |θ|", np.abs(angle).mean(), "°"),
        ("угол, максимум |θ|", np.abs(angle).max(), "°"),
        ("скачок угла между кадрами, среднее", angle_step.mean() if len(angle_step) else 0, "°"),
        ("скачок угла между кадрами, максимум", angle_step.max() if len(angle_step) else 0, "°"),
        ("скорость поворота", rate_per_hour(hours, angle), "°/ч"),
        ("радиус, среднее", col("radius").mean(), "px"),
        ("радиус, разброс (std)", col("radius").std(), "px"),
        ("время на кадр, среднее", col("ms").mean(), "мс"),
        ("время всего", elapsed, "с"),
    ]
    if "error" in records[0]:
        error = col("error")
        rows += [
            ("ошибка угла, среднее (смещение)", error.mean(), "°"),
            ("ошибка угла, среднее |e|", np.abs(error).mean(), "°"),
            ("ошибка угла, максимум |e|", np.abs(error).max(), "°"),
        ]
    return [{"metric": name,
             "value": value if isinstance(value, int) else round(float(value), 3),
             "unit": unit}
            for name, value, unit in rows]


def main():
    paths = list_frames(INPUT_DIR)
    if not paths:
        raise SystemExit(f"В папке {INPUT_DIR} нет кадров")
    if TEST_ANGLES is not None and len(TEST_ANGLES) != len(paths):
        raise SystemExit(f"Кадров {len(paths)}, а углов в TEST_ANGLES {len(TEST_ANGLES)}")

    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if CENTERED_DIR is not None:
        CENTERED_DIR.mkdir(parents=True, exist_ok=True)

    base_profiles = None
    base_radius = None
    base_angle = 0.0
    base_time = None
    records = []                 # метрики каждого кадра -> frames.csv
    start_all = time.perf_counter()

    for i, path in enumerate(paths):
        start = time.perf_counter()

        color = cv2.imread(str(path))
        if color is None:
            print(f"{path.name}: не удалось прочитать")
            continue

        if TEST_ANGLES is not None and TEST_ANGLES[i] != 0:
            fc = frame_center(color)
            color = rotate(color, TEST_ANGLES[i], fc, fc)

        # Центрирование
        result = center_frame(color)
        if result is None:
            print(f"{path.name}: обработка невозможна")
            continue
        centered, info = result
        sun_center = (info["after_x"], info["after_y"])

        if CENTERED_DIR is not None:
            out = CENTERED_DIR / f"{CENTERED_PREFIX}{path.stem}.png"
            cv2.imwrite(str(out), centered)

        # Угол относительно базы (база — первый годный кадр)
        gray = cv2.cvtColor(centered, cv2.COLOR_BGR2GRAY)
        if base_profiles is None:
            base_radius = info["radius"]
            base_profiles = ring_profiles(gray, (*sun_center, base_radius), N_POINTS, RINGS)
            if TEST_ANGLES is not None:
                base_angle = TEST_ANGLES[i]
            base_time = frame_time(path)
            theta = 0.0
        else:
            profiles = ring_profiles(gray, (*sun_center, base_radius), N_POINTS, RINGS)
            theta = pair_angle(base_profiles, profiles, N_POINTS)

        # Обратный поворот вокруг центра Солнца
        aligned = rotate(centered, -theta, sun_center, sun_center) if theta != 0 else centered
        cv2.imwrite(str(ALIGNED_DIR / f"{ALIGNED_PREFIX}{path.name}"), aligned)

        ms = (time.perf_counter() - start) * 1000

        # время от базы в часах (если в имени нет времени — номер кадра)
        shot = frame_time(path)
        hours = (shot - base_time).total_seconds() / 3600 if shot and base_time else float(i)

        # печать по кадру
        print(f"\n{path.name}")
        print(f"  цель:          ({info['target_x']:.2f}, {info['target_y']:.2f})")
        print(f"  было:          ({info['center_x']:.2f}, {info['center_y']:.2f})")
        print(f"  сдвиг:         dx={info['dx']:+d}, dy={info['dy']:+d}")
        print(f"  стало:         ({info['after_x']:.2f}, {info['after_y']:.2f})")
        line = f"  угол:          {theta:+.2f}°"
        if TEST_ANGLES is not None:
            expected = TEST_ANGLES[i] - base_angle
            line += f" (задано {expected:+.2f}°, ошибка {theta - expected:+.2f}°)"
        print(line)
        print(f"  время:         {ms:.0f} мс")

        # строка метрик кадра
        record = {
            "file": path.name,
            "hours": round(hours, 3),
            "target_x": info["target_x"], "target_y": info["target_y"],
            "center_x": round(info["center_x"], 3), "center_y": round(info["center_y"], 3),
            "dx": info["dx"], "dy": info["dy"],
            "after_x": round(info["after_x"], 3), "after_y": round(info["after_y"], 3),
            "radius": round(info["radius"], 2),
            "threshold": info["threshold"],
            "angle": theta,
            "ms": round(ms),
        }
        if TEST_ANGLES is not None:
            record["expected"] = expected
            record["error"] = round(theta - expected, 3)
        records.append(record)

    if not records:
        raise SystemExit("Ни один кадр не обработан")

    # общие метрики серии
    elapsed = time.perf_counter() - start_all
    summary = summarize(records, len(paths), elapsed)
    save_csv(FRAMES_CSV, records)
    save_csv(SUMMARY_CSV, summary)

    print("\n" + "=" * 60)
    print("Общие метрики серии")
    print("=" * 60)
    for row in summary:
        print(f"{row['metric']:<42} {row['value']:>10} {row['unit']}")

    print(f"\nКадры:   {ALIGNED_DIR}")
    print(f"Метрики: {FRAMES_CSV.name}, {SUMMARY_CSV.name} в {METRICS_DIR}")


if __name__ == "__main__":
    main()