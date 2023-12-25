import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def download_files_from_url(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith(('pdf', 'zip', 'docx', 'pptx')):  # 可以根据需要调整文件类型
            download_url = urljoin(url, href)
            filename = os.path.join(dest_folder, href.split('/')[-1])
            download_file(download_url, filename)

def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

url = 'http://home.ustc.edu.cn/~wx309/lecture/alg/index.html'  # 指定的URL
dest_folder = '/Users/pongyomo/Documents/ustc-course/算法设计与分析/'  # 替换为你的目标文件夹路径

download_files_from_url(url, dest_folder)
