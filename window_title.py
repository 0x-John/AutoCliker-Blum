import pygetwindow as gw

def list_blum_window_titles():
    windows = gw.getAllTitles()
    if not windows:
        print("Не найдено открытых окон.")
        return
    
    blum_windows = [title for title in windows if "Blum" in title]
    
    if not blum_windows:
        print("Не найдено окон с заголовком, содержащим 'Blum'.")
        return

    print("Заголовки окон, содержащие 'Blum':")
    for title in blum_windows:
        print(title)

if __name__ == "__main__":
    list_blum_window_titles()
