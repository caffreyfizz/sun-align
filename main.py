from solar_center import find_center_x, find_center_y, shift_image, find_otsu_threshold
from solar_rotation import rotate, ring_profiles, pair_angle
import cv2
import numpy as np
import os


color = cv2.imread("pictures/20260301_001741_aia304.jpg")                  # цветной кадр (h, w, 3)
image = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)          # серая копия для расчётов

os.makedirs("centered_pictures", exist_ok=True)

#----------------------------------image to center-------------------------------------#


center_x, center_y, t_best, _ = find_otsu_threshold(image)


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

cv2.imwrite("centered_pictures/image_centered.jpg", result)

'''
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
cv2.destroyAllWindows()'''




#----------------------------------------rotation------------------------------------------#


INPUT_DIR = "centered_pictures"      
OUTPUT_DIR = "aligned_pictures"

ANGLES = [0, 3.5, -5.89, 83, -2.9, 6.33, -9.4, 4.1, -7, 10.5, -4.75, 30.12]


RINGS = list(np.linspace(0.70, 0.95, 12)) + list(np.linspace(1.04, 1.20, 18))

N = 720                     # точек на кольце (шаг 0.5 градуса)
 

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
suns = []
for g in grays:
    c_x, c_y, _, radius = find_otsu_threshold(g)
    suns.append((c_x, c_y, radius))
radius = float(np.median([s[2] for s in suns]))
suns = [(s[0], s[1], radius) for s in suns]
 
# Найденный угол

profiles = [ring_profiles(g, s, N, RINGS) for g, s in zip(grays, suns)]


M = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        M[i, j] = pair_angle(profiles[i], profiles[j], N)
        M[j, i] = -M[i, j]
 
# Угол каждого кадра - среднее по всем парам.
# M[i, j] ~ theta_j - theta_i (на диагонали 0), поэтому среднее по i даёт theta_j - mean(theta);
# вычитаем значение первого кадра, чтобы theta_0 = 0.
theta = M.mean(axis=0)
theta = theta - theta[0]


print("\nкадр  задано  найдено  ошибка")

for i in range(n):

    aligned = rotate(rotated[i], -theta[i], suns[i][:2], frame_center)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "rotated_" + names[i]), rotated[i])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "aligned_" + names[i]), aligned)

    expected = ANGLES[i] - ANGLES[0]

    print(f"{i + 1:>4}  {expected:>+6.1f}  {theta[i]:>+7.2f}  {theta[i] - expected:>+6.2f}")


print(f"\nГотово: {n} кадров записано в папку {OUTPUT_DIR}")