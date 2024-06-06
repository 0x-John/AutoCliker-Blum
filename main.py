import math
import time
import random
import os

import cv2
import keyboard
import mss
import numpy as np
import pygetwindow as gw
import win32api
import win32con

# Название окна с игрой - Узнать можно запустив скрипт python window_title.py
WINDOW_TITLE = "Blum - SunBrowser"
# Процент целей для обработки - настраивайте под себя, зависит от размера вашего монитора, окна с игрой, производительности ПК и тд. Больше значение - больше звезд собираем.
TARGET_PERCENTAGE = 0.02
# Путь к файлу шаблона кнопки Play - Сделайте свой скриншот кнопки, если не происходит автоматический запуск игр после их окончания.
TEMPLATE_PATH = "template_play_button.png"
# Порог совпадения шаблона - Можете попробовать уменьшить это значение, если не происходит автоматический запуск игр после их окончания.
THRESHOLD = 0.8
# Интервал проверки кнопки "Play" в секундах
CHECK_INTERVAL = 5


class Logger:
    def __init__(self, prefix=None):
        self.prefix = prefix

    def log(self, data: str):
        if self.prefix:
            print(f"{self.prefix} {data}")
        else:
            print(data)


class AutoClicker:
    def __init__(self, window_title, target_colors_hex, nearby_colors_hex, template_path, threshold, logger):
        self.window_title = window_title
        self.target_colors_hex = target_colors_hex
        self.nearby_colors_hex = nearby_colors_hex
        self.template_path = template_path
        self.threshold = threshold
        self.logger = logger
        self.running = False
        self.clicked_points = []
        self.iteration_count = 0
        self.last_check_time = time.time()

        if not os.path.isfile(self.template_path):
            raise FileNotFoundError(f"Файл шаблона не найден: {self.template_path}")

    @staticmethod
    def hex_to_hsv(hex_color):
        hex_color = hex_color.lstrip('#')
        h_len = len(hex_color)
        rgb = tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))
        rgb_normalized = np.array([[rgb]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2HSV)
        return hsv[0][0]

    @staticmethod
    def click_at(x, y):
        try:
            if not (0 <= x < win32api.GetSystemMetrics(0) and 0 <= y < win32api.GetSystemMetrics(1)):
                raise ValueError(f"Координаты вне пределов экрана: ({x}, {y})")
            win32api.SetCursorPos((x, y))
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
        except Exception as e:
            print(f"Ошибка при установке позиции курсора: {e}")

    def toggle_script(self):
        self.running = not self.running
        r_text = "вкл" if self.running else "выкл"
        self.logger.log(f'Статус изменен: {r_text}')

    def is_near_color(self, hsv_img, center, target_hsvs, radius=8):
        x, y = center
        height, width = hsv_img.shape[:2]
        for i in range(max(0, x - radius), min(width, x + radius + 1)):
            for j in range(max(0, y - radius), min(height, y + radius + 1)):
                distance = math.sqrt((x - i) ** 2 + (y - j) ** 2)
                if distance <= radius:
                    pixel_hsv = hsv_img[j, i]
                    for target_hsv in target_hsvs:
                        if np.allclose(pixel_hsv, target_hsv, atol=[1, 50, 50]):
                            return True
        return False

    def check_and_click_play_button(self, sct, monitor):
        current_time = time.time()
        if current_time - self.last_check_time >= CHECK_INTERVAL:
            self.last_check_time = current_time
            template = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                self.logger.log(f"Не удалось загрузить файл шаблона: {self.template_path}")
                return

            template_height, template_width = template.shape

            img = np.array(sct.grab(monitor))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= self.threshold)

            matched_points = list(zip(*loc[::-1]))

            if matched_points:
                pt_x, pt_y = matched_points[0]
                cX = pt_x + template_width // 2 + monitor["left"]
                cY = pt_y + template_height // 2 + monitor["top"]

                self.click_at(cX, cY)
                self.logger.log(f'Нажал на кнопку "Play": {cX} {cY}')
                self.clicked_points.append((cX, cY))

    def click_color_areas(self):
        windows = [win for win in gw.getAllTitles() if win == self.window_title]
        if not windows:
            self.logger.log(
                f"Не найдено окна с заголовком: {self.window_title}. Откройте Веб-приложение Blum и откройте скрипт заново")
            return

        if len(windows) > 1:
            self.logger.log(f"Обнаружено несколько окон с заголовком: {self.window_title}. Оставьте только одно окно с игрой.")
            return

        window = gw.getWindowsWithTitle(windows[0])[0]
        window.activate()
        target_hsvs = [self.hex_to_hsv(color) for color in self.target_colors_hex]
        nearby_hsvs = [self.hex_to_hsv(color) for color in self.nearby_colors_hex]

        with mss.mss() as sct:
            keyboard.add_hotkey('F6', self.toggle_script)

            while True:
                if self.running:
                    monitor = {
                        "top": window.top,
                        "left": window.left,
                        "width": window.width,
                        "height": window.height
                    }
                    img = np.array(sct.grab(monitor))
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

                    for target_hsv in target_hsvs:
                        lower_bound = np.array([max(0, target_hsv[0] - 1), 30, 30])
                        upper_bound = np.array([min(179, target_hsv[0] + 1), 255, 255])
                        mask = cv2.inRange(hsv, lower_bound, upper_bound)
                        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        num_contours = len(contours)
                        num_to_click = int(num_contours * TARGET_PERCENTAGE)
                        contours_to_click = random.sample(contours, num_to_click)

                        for contour in reversed(contours_to_click):
                            if cv2.contourArea(contour) < 1:
                                continue

                            M = cv2.moments(contour)
                            if M["m00"] == 0:
                                continue
                            cX = int(M["m10"] / M["m00"]) + monitor["left"]
                            cY = int(M["m01"] / M["m00"]) + monitor["top"]

                            if not self.is_near_color(hsv, (cX - monitor["left"], cY - monitor["top"]), nearby_hsvs):
                                continue

                            if any(math.sqrt((cX - px) ** 2 + (cY - py) ** 2) < 35 for px, py in self.clicked_points):
                                continue
                            cY += 5
                            self.click_at(cX, cY)
                            self.logger.log(f'Нажал: {cX} {cY}')
                            self.clicked_points.append((cX, cY))

                    self.check_and_click_play_button(sct, monitor)
                    time.sleep(0.1)
                    self.iteration_count += 1
                    if self.iteration_count >= 5:
                        self.clicked_points.clear()
                        self.iteration_count = 0


if __name__ == "__main__":
    # Убедитесь, что текущая рабочая директория соответствует директории скрипта
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    logger = Logger("[https://t.me/x_0xJohn]")
    logger.log("Вас приветствует бесплатный скрипт - автокликер для игры Blum")
    logger.log('После запуска мини игры нажимайте клавишу F6 на клавиатуре')
    target_colors_hex = ["#c9e100", "#bae70e"]
    nearby_colors_hex = ["#abff61", "#87ff27"]
    template_path = os.path.join(current_dir, TEMPLATE_PATH)
    threshold = THRESHOLD  # Порог совпадения шаблона

    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Файл шаблона не найден: {template_path}")

    auto_clicker = AutoClicker(WINDOW_TITLE, target_colors_hex, nearby_colors_hex, template_path, threshold, logger)
    try:
        auto_clicker.click_color_areas()
    except Exception as e:
        logger.log(f"Произошла ошибка: {e}")
    for i in reversed(range(5)):
        i += 1
        print(f"Скрипт завершит работу через {i}")
        time.sleep(1)
