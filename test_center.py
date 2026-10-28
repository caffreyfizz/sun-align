import cv2
import numpy as np

color = cv2.imread("pictures/image.png")                  # цветной кадр (h, w, 3)
image = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)          # серая копия для расчётов


def find_center_x(img, t, step=8):
    mids = []
    lengths = []
    last = img.shape[1] - 1
    prev_cut = False

    for y in range(0, img.shape[0], step):
        row = img[y].tolist()

        # хорда упирается в левый или правый край кадра
        if row[0] > t or row[last] > t:
            if prev_cut:
                side = "слева" if row[0] > t else "справа"
                print("обработка невозможна: солнце за краем", side)
                return None
            prev_cut = True
            continue
        prev_cut = False

        left = None
        for i in range(1, last + 1):
            if row[i] > t:
                left = (i - 1) + (t - row[i - 1]) / (row[i] - row[i - 1])
                break
        if left is None:
            continue

        for i in range(last - 1, -1, -1):
            if row[i] > t:
                right = i + (t - row[i]) / (row[i + 1] - row[i])
                break

        mids.append((left + right) / 2)
        lengths.append(right - left)

    if not lengths:
        print("обработка невозможна: солнце не найдено")
        return None

    half = max(lengths) / 2
    good = []
    for k in range(len(mids)):
        if lengths[k] > half:
            good.append(mids[k])
    return good


def find_center_y(img, t, step=8):
    mids = []
    lengths = []
    last = img.shape[0] - 1
    prev_cut = False

    for x in range(0, img.shape[1], step):
        col = img[:, x].tolist()

        # хорда упирается в верхний или нижний край кадра
        if col[0] > t or col[last] > t:
            if prev_cut:
                side = "сверху" if col[0] > t else "снизу"
                print("обработка невозможна: солнце за краем", side)
                return None
            prev_cut = True
            continue
        prev_cut = False

        top = None
        for i in range(1, last + 1):
            if col[i] > t:
                top = (i - 1) + (t - col[i - 1]) / (col[i] - col[i - 1])
                break
        if top is None:
            continue

        for i in range(last - 1, -1, -1):
            if col[i] > t:
                bottom = i + (t - col[i]) / (col[i + 1] - col[i])
                break

        mids.append((top + bottom) / 2)
        lengths.append(bottom - top)

    if not lengths:
        print("обработка невозможна: солнце не найдено")
        return None

    half = max(lengths) / 2
    good = []
    for k in range(len(mids)):
        if lengths[k] > half:
            good.append(mids[k])
    return good


# Гистограмма: сколько пикселей каждой яркости
hist = [0] * 256
for row in image:
    for v in row.tolist():
        hist[v] += 1

# Общие величины
N = 0
S = 0
for v in range(256):
    N += hist[v]
    S += v * hist[v]

''' σ²(t) = w₀ · w₁ · (μ₀ − μ₁)²  - Перебор порогов
w₀, w₁ — доля пикселей в каждом классе;
μ₀, μ₁ — средняя яркость каждого класса;
(μ₀ - μ₁)² — насколько далеко друг от друга средние;
w₀ · w₁ — штраф за перекос'''

n0 = 0    #сколько пикселей в классе 0
s0 = 0    # суммарная якрость пикселей в классе 0
best_sigma = -1.0
t_low = 0
t_high = 0

for t in range(256):
    n0 += hist[t]
    s0 += t * hist[t]
    n1 = N - n0

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
if medians_x is None:
    exit()
center_x = sorted(medians_x)[len(medians_x) // 2]

medians_y = find_center_y(image, t_best)
if medians_y is None:
    exit()
center_y = sorted(medians_y)[len(medians_y) // 2]


def shift_image(img, dx, dy):
    height, width = img.shape[:2]
    result = np.zeros_like(img)
    if abs(dx) >= width or abs(dy) >= height:
        return result

    # какие строки и столбцы исходного кадра попадут в новый
    src_y0 = max(0, -dy)    # первый индекс исходника, который попадёт в новый кадр
    src_y1 = min(height, height - dy)     # последний индекс исходника, который попадёт в новый кадр
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)

    result[src_y0 + dy: src_y1 + dy, src_x0 + dx: src_x1 + dx] = img[src_y0:src_y1, src_x0:src_x1]
    return result


height, width = image.shape
target_x = (width - 1) / 2
target_y = (height - 1) / 2

dx = round(target_x - center_x)
dy = round(target_y - center_y)
result = shift_image(color, dx, dy)
print("сдвиг:", dx, dy)

# проверка: где центр после сдвига
result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
new_medians_x = find_center_x(result_gray, t_best)
new_medians_y = find_center_y(result_gray, t_best)

new_x = sorted(new_medians_x)[len(new_medians_x) // 2]
new_y = sorted(new_medians_y)[len(new_medians_y) // 2]

print("стало:", new_x, new_y, "цель:", target_x, target_y)

cv2.imwrite("pictures/image_centered.png", result)


#отображение
left = color.copy()
right = result.copy()

cx_mark, cy_mark = int(round(target_x)), int(round(target_y))
for pic in (left, right):
    cv2.line(pic, (cx_mark - 25, cy_mark), (cx_mark + 25, cy_mark), (0, 0, 255), 2)
    cv2.line(pic, (cx_mark, cy_mark - 25), (cx_mark, cy_mark + 25), (0, 0, 255), 2)

view = np.hstack((left, right))
view = cv2.resize(view, (view.shape[1] // 2, view.shape[0] // 2))

cv2.imshow("left: original, right: centered", view)
cv2.waitKey(0)
cv2.destroyAllWindows()