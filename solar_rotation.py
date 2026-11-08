import numpy as np
from solar_center import find_center_x, find_center_y


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


def cubic_weights(t, a=-0.75):
    # Веса 4 соседних пикселей (слева-1, слева, справа, справа+1) для дробной части t.
    t2 = t * t
    t3 = t * t * t
    return [a * t3 - 2 * a * t2 + a * t,
            (a + 2) * t3 - (a + 3) * t2 + 1,
            -(a + 2) * t3 + (2 * a + 3) * t2 - a * t,
            -a * t3 + a * t2]
 
 
def sample_cubic(img, xs, ys):
    # 16 соседних пикселей с кубическими весами.
    h, w = img.shape[:2]

    inside = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)

    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)

    wx = cubic_weights(xs - x0)
    wy = cubic_weights(ys - y0)

 
    if img.ndim == 3:
        wx = [v[..., None] for v in wx]
        wy = [v[..., None] for v in wy]
        inside = inside[..., None]
 
    result = 0.0
    for j in range(4):
        yi = np.clip(y0 - 1 + j, 0, h - 1)
        row = 0.0
        for i in range(4):
            xi = np.clip(x0 - 1 + i, 0, w - 1)
            row = row + wx[i] * img[yi, xi]

        result = result + wy[j] * row
 
    return np.where(inside, result, 0)
 
# Поворот кадра
def rotate(img, angle, center, new_center):

    h, w = img.shape[:2]
    a = np.radians(angle)

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    u = x - new_center[0]                             # новые пиксели со смещением по Х
    v = -(y - new_center[1])                            #  новые пискели со смещением по Y      

    src_x = center[0] + u * np.cos(a) + v * np.sin(a)     # src_y = center[0] + u_src
    src_y = center[1] - (-u * np.sin(a) + v * np.cos(a))  # src_y = center[1] − v_src

    return sample_cubic(img, src_x, src_y).round().clip(0, 255).astype(np.uint8)
 


# Создание профилей для линейной корреляции

def ring_profiles(gray, sun, N, RINGS):

    cx, cy, radius = sun
    phi = 2 * np.pi * np.arange(N) / N
    profiles = []
    for f in RINGS:
        p = sample_cubic(np.log1p(gray.astype(np.float64)),
                   cx + f * radius * np.cos(phi), cy - f * radius * np.sin(phi))

        # Центрируем данные
        p = p - p.mean()
    
        profiles.append(p / (p.std() + 1e-9))

    return np.array(profiles)
 
#Циклическая корреляция base c каждым изображением 

def pair_angle(base, image, N):
    
    corr = np.zeros(N)
    for s in range(N):               # корреляция при циклическом сдвиге на s точек
        corr[s] = corr[s] = np.sum(np.sqrt((image - np.roll(base, s, axis=1))**2))

    s = int(np.argmin(corr))


    if s > N / 2:
        s -= N
        return s * 360 / N