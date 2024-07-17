import os
import time
import random
import math
import cv2
import keyboard
import mss
import numpy as np
import pygetwindow as gw
import win32api
import win32con
import warnings
from pywinauto import Application
import config

warnings.filterwarnings("ignore", category=UserWarning, module='pywinauto')

def list_windows_by_title(title_keywords):
    windows = gw.getAllWindows()
    filtered_windows = []
    for window in windows:
        for keyword in title_keywords:
            if keyword.lower() in window.title.lower():
                filtered_windows.append((window.title, window._hWnd))
                break
    return filtered_windows

class Logger:
    def __init__(self, prefix=None):
        self.prefix = prefix

    def log(self, data: str):
        if self.prefix:
            print(f"{self.prefix} {data}")
        else:
            print(data)

class AutoClicker:
    def __init__(self, hwnd, target_colors_hex, nearby_colors_hex, threshold, logger, target_percentage, collect_freeze):
        self.hwnd = hwnd
        self.target_colors_hex = target_colors_hex
        self.nearby_colors_hex = nearby_colors_hex
        self.threshold = threshold
        self.logger = logger
        self.target_percentage = target_percentage
        self.collect_freeze = collect_freeze
        self.running = False
        self.clicked_points = []
        self.iteration_count = 0
        self.last_check_time = time.time()
        self.last_freeze_check_time = time.time()
        self.freeze_cooldown_time = 0
        self.game_start_time = None
        self.freeze_count = 0
        self.target_hsvs = [self.hex_to_hsv(color) for color in self.target_colors_hex]
        self.nearby_hsvs = [self.hex_to_hsv(color) for color in self.nearby_colors_hex]

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
        if self.running:
            self.game_start_time = None
            self.freeze_count = 0
            self.logger.log('Скрипт запущен. Ищем кнопку начала игры.')
        else:
            self.logger.log('Скрипт остановлен.')

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
        if current_time - self.last_check_time >= random.uniform(config.CHECK_INTERVAL_MIN, config.CHECK_INTERVAL_MAX):
            self.last_check_time = current_time
            templates = [
                cv2.imread(os.path.join("template_png", "template_play_button.png"), cv2.IMREAD_GRAYSCALE),
                cv2.imread(os.path.join("template_png", "template_play_button1.png"), cv2.IMREAD_GRAYSCALE)
            ]

            for template in templates:
                if template is None:
                    self.logger.log("Не удалось загрузить файл шаблона.")
                    continue

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
                    self.logger.log(f'Нажал на кнопку: {cX} {cY}')
                    self.clicked_points.append((cX, cY))
                    self.game_start_time = time.time()
                    self.freeze_count = 0  # Сбросить счетчик заморозок при начале новой игры
                    break  # Остановить проверку после первого найденного совпадения

    def click_color_areas(self):
        app = Application().connect(handle=self.hwnd)
        window = app.window(handle=self.hwnd)
        window.set_focus()

        with mss.mss() as sct:
            keyboard.add_hotkey(config.HOTKEY, self.toggle_script)
            self.logger.log(f'Нажмите {config.HOTKEY} для запуска/остановки скрипта.')

            while True:
                if self.running:
                    rect = window.rectangle()
                    monitor = {
                        "top": rect.top,
                        "left": rect.left,
                        "width": rect.width(),
                        "height": rect.height()
                    }
                    img = np.array(sct.grab(monitor))
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

                    if self.game_start_time is None:
                        self.check_and_click_play_button(sct, monitor)
                    elif self.is_game_over():
                        self.logger.log('Игра окончена.')
                        self.random_delay_before_restart()
                        self.game_start_time = None
                    else:
                        self.click_on_targets(hsv, monitor, sct)
                time.sleep(0.1)

    def is_game_over(self):
        game_duration = 30 + self.freeze_count * 3
        current_time = time.time()
        if self.game_start_time and current_time - self.game_start_time >= game_duration - 0.5:
            return True
        return False

    def click_on_targets(self, hsv, monitor, sct):
        for target_hsv in self.target_hsvs:
            lower_bound = np.array([max(0, target_hsv[0] - 1), 30, 30])
            upper_bound = np.array([min(179, target_hsv[0] + 1), 255, 255])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            num_contours = len(contours)
            num_to_click = int(num_contours * self.target_percentage)
            contours_to_click = random.sample(contours, num_to_click)

            for contour in reversed(contours_to_click):
                if cv2.contourArea(contour) < 6:
                    continue

                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cX = int(M["m10"] / M["m00"]) + monitor["left"]
                cY = int(M["m01"] / M["m00"]) + monitor["top"]

                if not self.is_near_color(hsv, (cX - monitor["left"], cY - monitor["top"]), self.nearby_hsvs):
                    continue

                if any(math.sqrt((cX - px) ** 2 + (cY - py) ** 2) < 35 for px, py in self.clicked_points):
                    continue
                cY += 5
                self.click_at(cX, cY)
                self.logger.log(f'Нажал: {cX} {cY}')
                self.clicked_points.append((cX, cY))

        if self.collect_freeze:
            self.check_and_click_freeze_button(sct, monitor)
        
        self.iteration_count += 1
        if self.iteration_count >= 5:
            self.clicked_points.clear()
            self.iteration_count = 0

    def check_and_click_freeze_button(self, sct, monitor):
        freeze_hsvs = [self.hex_to_hsv(color) for color in config.FREEZE_COLORS_HEX]
        current_time = time.time()
        if current_time - self.last_freeze_check_time >= 1 and current_time >= self.freeze_cooldown_time:
            self.last_freeze_check_time = current_time
            img = np.array(sct.grab(monitor))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            for freeze_hsv in freeze_hsvs:
                lower_bound = np.array([max(0, freeze_hsv[0] - 1), 30, 30])
                upper_bound = np.array([min(179, freeze_hsv[0] + 1), 255, 255])
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) < 3:
                        continue

                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cX = int(M["m10"] / M["m00"]) + monitor["left"]
                    cY = int(M["m01"] / M["m00"]) + monitor["top"]

                    self.click_at(cX, cY)
                    self.logger.log(f'Нажал на заморозку: {cX} {cY}')
                    self.freeze_cooldown_time = time.time() + 4  # Установить паузу на 4 секунды для поиска заморозок
                    self.freeze_count += 1

                    # Проверка цвета пикселя через 1 секунду после клика
                    time.sleep(1)

                    img_check = np.array(sct.grab(monitor))
                    img_bgr_check = cv2.cvtColor(img_check, cv2.COLOR_BGRA2BGR)
                    hsv_check = cv2.cvtColor(img_bgr_check, cv2.COLOR_BGR2HSV)

                    right_bottom_x = monitor["width"] - config.OFFSET_X
                    right_bottom_y = monitor["height"] - config.OFFSET_Y

                    if right_bottom_x >= img_check.shape[1] or right_bottom_y >= img_check.shape[0]:
                        self.logger.log('Ошибка: правый нижний угол выходит за пределы изображения')
                        return

                    pixel_hsv = hsv_check[right_bottom_y, right_bottom_x]

                    # Вывод цвета пикселя в консоль
                    # self.logger.log(f'Цвет пикселя в правом нижнем углу: {pixel_hsv}')

                    # Подсветка пикселя (закомментировано, включить для отладки)
                    # cv2.circle(img_bgr_check, (right_bottom_x, right_bottom_y), 5, (0, 0, 255), 2)
                    # cv2.imwrite('pixel_check.png', img_bgr_check)

                    # Проверка на черный цвет
                    if np.array_equal(pixel_hsv, [0, 0, 0]):
                        self.freeze_count -= 1
                        self.logger.log('Ошибка: ложный клик по заморозке')

                    return

    def random_delay_before_restart(self):
        delay = random.uniform(config.CHECK_INTERVAL_MIN, config.CHECK_INTERVAL_MAX)
        self.logger.log(f'Задержка перед перезапуском: {delay:.2f} секунд.')
        time.sleep(delay)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    windows = list_windows_by_title(config.KEYWORDS)

    if not windows:
        print("Нет окон, содержащих указанные ключевые слова Blum или Telegram.")
        exit()

    print("Доступные окна для выбора:")
    for i, (title, hwnd) in enumerate(windows):
        print(f"{i + 1}: {title}")

    choice = int(input("Введите номер окна, в котором открыт бот Blum: ")) - 1
    if choice < 0 or choice >= len(windows):
        print("Неверный выбор.")
        exit()

    hwnd = windows[choice][1]

    while True:
        try:
            target_percentage = input("Введите значение от 0 до 1 для рандомизации прокликивания звезд, где 1 означает сбор всех звезд. (Выбор величины зависит от множества факторов: размера экрана, окна и т.д.) Я выбираю значения 0.04 - 0.06 для сбора около 140-150 звезд. Вам необходимо самостоятельно подобрать необходимое значение: ")
            target_percentage = target_percentage.replace(',', '.')
            target_percentage = float(target_percentage)
            if 0 <= target_percentage <= 1:
                break
            else:
                print("Пожалуйста, введите значение от 0 до 1.")
        except ValueError:
            print("Неверный формат. Пожалуйста, введите число.")

    while True:
        try:
            collect_freeze = int(input("Кликать заморозку? 1 - ДА, 2 - НЕТ: "))
            if collect_freeze in [1, 2]:
                collect_freeze = (collect_freeze == 1)
                break
            else:
                print("Пожалуйста, введите 1 или 2.")
        except ValueError:
            print("Неверный формат. Пожалуйста, введите число.")

    logger = Logger("[https://t.me/x_0xJohn]")
    logger.log("Вас приветствует бесплатный скрипт - автокликер для игры Blum")

    auto_clicker = AutoClicker(hwnd, config.TARGET_COLORS_HEX, config.NEARBY_COLORS_HEX, config.THRESHOLD, logger, target_percentage, collect_freeze)
    try:
        auto_clicker.click_color_areas()
    except Exception as e:
        logger.log(f"Произошла ошибка: {e}")
    for i in reversed(range(5)):
        print(f"Скрипт завершит работу через {i}")
        time.sleep(1)
