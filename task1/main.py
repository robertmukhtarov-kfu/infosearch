import os
from pathlib import Path
from urllib import request

# Получение массива ссылок из предварительно подготовленного списка urls.txt
with open('task1/urls.txt') as urls_file:
    urls = [line.rstrip() for line in urls_file]

# Создание папки downloads, в которую будут выкачиваться страницы
# При каждом запуске программы папка очищается
downloads_dir = 'task1/downloads'
Path(downloads_dir).mkdir(parents=True, exist_ok=True)
for file in os.listdir(downloads_dir):
    os.remove(os.path.join(downloads_dir, file))

# Скачивание, запись страницы в текстовый файл; добавление номера и ссылки в index.txt
for i, url in enumerate(urls):
    try:
        html_page = request.urlopen(url).read().decode()
    except Exception as e:
        print(f'Failed to download {url}: {e.args[0]}')
    else:
        with open(f'{downloads_dir}/{i + 1}.txt', 'w') as html_file:
            html_file.write(html_page)
        with open('task1/index.txt', 'a') as index_file:
            index_file.write(f'{i + 1}. {url}\n')
        print(f'Downloaded {url}')
