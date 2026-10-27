import os
 
import cv2
import numpy as np
 
INPUT_DIR = "Solar_figna/2026_03"      
OUTPUT_DIR = "Solar_figna/2026_03_result"

ANGLES = [0, 3.5, -5.89, 8.3, -2.9, 6.33, -9.4, 4.1, -7, 10.5, -4.75, 2.12]


RINGS = list(np.linspace(0.70, 0.95, 12)) + list(np.linspace(1.04, 1.20, 18))

N = 720                     # точек на кольце (шаг 0.5 градуса)
 

def find_center_x(img, t, step=8):
    mids = []
    lengths = []
    for y in range(0, len(img), step):
        row = img[y].tolist()
 
        left = None
        for i in range(1, len(row)):
            if row[i] > t:
                left = (i - 1) + (t - row[i - 1]) / (row[i] - row[i - 1])
                break
        if left is None:
            continue
 
        right = None
        for i in range(len(row) - 2, -1, -1):
            if row[i] > t:
                right = i + (t - row[i]) / (row[i + 1] - row[i])
                break
 
        mids.append((left + right) / 2)
        lengths.append(right - left)
 
    longest = max(lengths)
    good = []
    for k in range(len(mids)):
        if lengths[k] > longest / 2:
            good.append(mids[k])
    return good
 
 
def find_center_y(img, t, step=8):
    mids = []
    lengths = []
    for x in range(0, len(img[0]), step):
        col = img[:, x].tolist()
 
        top = None
        for i in range(1, len(col)):
            if col[i] > t:
                top = (i - 1) + (t - col[i - 1]) / (col[i] - col[i - 1])
                break
        if top is None:
            continue
 
        bottom = None
        for i in range(len(col) - 2, -1, -1):
            if col[i] > t:
                bottom = i + (t - col[i]) / (col[i + 1] - col[i])
                break
 
        mids.append((top + bottom) / 2)
        lengths.append(bottom - top)
 
    longest = max(lengths)
    good = []
    for k in range(len(mids)):
        if lengths[k] > longest / 2:
            good.append(mids[k])
    return good
 
 
def find_sun(image):
    """Центр (x, y) и радиус Солнца по серому кадру."""
    # Гистограмма: сколько пикселей каждой яркости
    hist = [0] * 256
    for row in image:
        for v in row.tolist():
            hist[v] += 1
 
    # Общие величины
    total = 0
    S = 0
    for v in range(256):
        total += hist[v]
        S += v * hist[v]
 
    # Порог Оцу: sigma(t) = n0 * n1 * (q0 - q1)^2, перебор порогов
    n0 = 0
    s0 = 0
    best_sigma = -1.0
    t_low = 0
    t_high = 0
    for t in range(256):
        n0 += hist[t]
        s0 += t * hist[t]
        n1 = total - n0
        if n0 == 0:
            continue
        if n1 == 0:
            break
        q0 = s0 / n0
        q1 = (S - s0) / n1
        sigma = n0 * n1 * (q0 - q1) ** 2
        if sigma > best_sigma:
            best_sigma = sigma
            t_low = t
            t_high = t
        elif sigma == best_sigma:
            t_high = t
    t_best = (t_low + t_high) // 2
 
    medians_x = find_center_x(image, t_best)
    center_x = sorted(medians_x)[len(medians_x) // 2]
    medians_y = find_center_y(image, t_best)
    center_y = sorted(medians_y)[len(medians_y) // 2]
 
    # радиус: из площади диска (число пикселей ярче порога)
    radius = (np.count_nonzero(image > t_best) / np.pi) ** 0.5
    return center_x, center_y, radius
 

# Интерполяция

def sample(img, xs, ys):

    h, w = img.shape[:2]
    inside = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 2)
    fx = np.clip(xs - x0, 0, 1)                              #вес правого соседа
    fy = np.clip(ys - y0, 0, 1)                                   #вес верхнего соседа

    if img.ndim == 3:
        fx, fy, inside = fx[..., None], fy[..., None], inside[..., None]


    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx

    bottom = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx

    result = top * (1 - fy) + bottom * fy

    return np.where(inside, result , 0)
 
# Поворот кадра
def rotate(img, angle, center, new_center):

    h, w = img.shape[:2]
    a = np.radians(angle)

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    u = x - new_center[0]                             # новые пиксели со смещением по Х
    v = -(y - new_center[1])                            #  новые пискели со смещением по Y      

    src_x = center[0] + u * np.cos(a) + v * np.sin(a)     # src_y = center[0] + u_src
    src_y = center[1] - (-u * np.sin(a) + v * np.cos(a))  # src_y = center[1] − v_src

    return sample(img, src_x, src_y).round().clip(0, 255).astype(np.uint8)
 


# Создание профилей для линейной корреляции

def ring_profiles(gray, sun):

    cx, cy, radius = sun
    phi = 2 * np.pi * np.arange(N) / N
    profiles = []
    for f in RINGS:
        p = sample(np.log1p(gray.astype(np.float64)),
                   cx + f * radius * np.cos(phi), cy - f * radius * np.sin(phi))

        # Центрируем данные
        p = p - p.mean()
    
        profiles.append(p / (p.std() + 1e-9))

    return np.array(profiles)
 
#Циклическая корреляция base c каждым изображением 

def pair_angle(base, image):
    
    corr = np.zeros(N)
    for s in range(N):               # корреляция при циклическом сдвиге на s точек
        corr[s] = (image * np.roll(base, s, axis=1)).mean()

    s = int(np.argmax(corr))


    if s > N / 2:
        s -= N

    return s * 360 / N
 

# Основная часть

names = sorted(os.listdir(INPUT_DIR))
n = len(names)
if n != len(ANGLES):
    raise SystemExit(f"В папке {n} кадров, а в ANGLES {len(ANGLES)} углов")
 
# Читаем и поворачиваем вокруг центра кадра
frames = [cv2.imread(os.path.join(INPUT_DIR, name)) for name in names]
h, w = frames[0].shape[:2]
frame_center = ((w - 1) / 2, (h - 1) / 2)
rotated = [rotate(f, a, frame_center, frame_center) if a != 0 else f
           for f, a in zip(frames, ANGLES)]
 
# Для всех кадров находим Солнце, делаем общий радиус
grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in rotated]
suns = [find_sun(g) for g in grays]
radius = float(np.median([s[2] for s in suns]))
suns = [(s[0], s[1], radius) for s in suns]
 
# Найденный угол

profiles = [ring_profiles(g, s) for g, s in zip(grays, suns)]

# M = np.zeros((n, n))

# for i in range(n):
#     for j in range(i + 1, n):
#         M[i, j] = pair_angle(profiles[i], profiles[j])
#         M[j, i] = -M[i, j]


# theta = M.mean(axis=0)

theta = np.zeros(n)
theta[0] = 0
for i in range(1, n):
    theta[i] = pair_angle(profiles[0], profiles[i])

theta = theta - theta[0]


print("\nкадр  задано  найдено  ошибка")
for i in range(n):

    aligned = rotate(rotated[i], -theta[i], suns[i][:2], frame_center)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "rotated_" + names[i]), rotated[i])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "aligned_" + names[i]), aligned)

    expected = ANGLES[i] - ANGLES[0]

    print(f"{i + 1:>4}  {expected:>+6.1f}  {theta[i]:>+7.2f}  {theta[i] - expected:>+6.2f}")

print(f"\nГотово: {n} кадров записано в папку {OUTPUT_DIR}")