import requests
from bs4 import BeautifulSoup
import csv
import re

# URL начальной страницы
base_url = 'https://www.biblio-globus.ru/category?cid=182&pagenumber='

headers = {
    "Accept": "*/*",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
}

with open("index.html") as file:
    src = file.read()

soup = BeautifulSoup(src, "lxml")
categories_hrefs = soup.find_all(class_ ="nav-link parent_category")

all_categories_dict = {}

for item in categories_hrefs:
    item_text = item.text
    item_href = "https://www.biblio-globus.ru/" + item.get("href")
    all_categories_dict[item_text] = item_href


# Базовый URL для формирования полных ссылок
BASE_URL = 'https://www.biblio-globus.ru//catalog/index/'

# Функция для получения объекта BeautifulSoup
def get_soup(url):
    response = requests.get(url)
    response.raise_for_status()  # Проверяем успешность запроса
    return BeautifulSoup(response.text, 'html.parser')

# Функция для извлечения общего количества страниц в субкатегории
def get_total_pages(soup):
    pagination = soup.find('ul', class_='pagination justify-content-center')
    if not pagination:
        return 1  # Если пагинации нет, возвращаем 1

    # Ищем ссылку с символами '»»', что соответствует переходу на последнюю страницу
    links = [a.get('href') for a in pagination.find_all('a', href=True) if 'page=' in a.get('href') ]
    if links:
        href = links[-1]
        if href:
            pattern = r"&page=(\d+)&sort"
            # Поиск совпадения
            match = re.search(pattern, href)
            if match:
                total_pages = int(match.group(1))
                return total_pages
    else:
        return 1

# Функция для парсинга страницы категории и извлечения субкатегорий
def parse_category_page(category_url):
    soup = get_soup(category_url)
    subcategories = {}

    # Находим ul с id 'catalogue'
    catalogue_ul = soup.find('ul', id='catalogue')
    if not catalogue_ul:
        return subcategories  # Если ul не найден, возвращаем пустой словарь

    # Проходим по всем непосредственным li внутри catalogue_ul
    for li in catalogue_ul.find_all('li', recursive=False):
        direct_a_tag = li.find('a', recursive=False)
        if direct_a_tag:
            subcategory_name = direct_a_tag.text.strip()
            href = direct_a_tag.get('href').strip()
            subcategory_url = BASE_URL + href
            subcategories[subcategory_name] = subcategory_url
        else:
            nested_ul = li.find('ul')
            if nested_ul:
                for sub_li in nested_ul.find_all('li'):
                    sub_a_tag = sub_li.find('a')
                    if sub_a_tag:
                        subcategory_name = sub_a_tag.text.strip()
                        href = sub_a_tag.get('href').strip()
                        subcategory_url = BASE_URL + href
                        subcategories[subcategory_name] = subcategory_url

    return subcategories

# Модифицированная функция для извлечения ссылок на страницы книг из всех страниц субкатегории
def parse_subcategory_page(subcategory_url):
    book_links = []
    
    # Извлечение идентификатора категории из subcategory_url
    category_id_match = re.search(r'/(\d+)$', subcategory_url)
    if category_id_match:
        category_id = category_id_match.group(1)
    else:
        raise ValueError("Не удалось извлечь идентификатор категории из URL")

    soup = get_soup(subcategory_url)
    total_pages = get_total_pages(soup)

    print(total_pages)
    
    for page_number in range(1, total_pages + 1):
        page_url = f"{BASE_URL[:-6]}category?id={category_id}&page={page_number}&sort=0&instock=&isdiscount="
        print(f"Парсинг страницы: {page_url}")  # Для демонстрации

        soup = get_soup(page_url)

        # Находим все div с классом 'product' на текущей странице
        product_divs = soup.find_all('div', class_='product')
        for product_div in product_divs:
            a_tag = product_div.find('a', class_='img_link')
            if a_tag:
                href = a_tag.get('href').strip()
                book_url = BASE_URL[:-15] + href
                book_links.append(book_url)

    return book_links

# Функция для извлечения информации о книге со страницы книги
def parse_book_page(book_url):
    soup = get_soup(book_url)

    # Извлечение URL изображения
    image_div = soup.find('div', class_='col-sm-12 col-md-12 col-lg-3')
    image_url = None
    if image_div:
        a_tag = image_div.find('a', {'data-fancybox': 'gallery'})
        if a_tag:
            image_url = a_tag.get('href')

    # Извлечение аннотации
    annotation_div = soup.find('div', id='collapseExample')
    annotation = annotation_div.get_text(strip=True) if annotation_div else None

    # Извлечение автора и названия книги из атрибута alt изображения
    img_tag = image_div.find('img') if image_div else None
    title_author = img_tag['alt'] if img_tag and 'alt' in img_tag.attrs else ''
    
    # Разделение на автора и название книги (если формат известен)
    title, author = title_author.split('«')[1].split('»') if '«' in title_author and '»' in title_author else ('', '')

    return {
        'page_url': book_url,
        'image_url': image_url,
        'author': author,
        'title': title,
        'annotation': annotation
    }

# Пример использования функций для сбора данных и записи в CSV
if __name__ == "__main__":
    # Проходим по всем категориям
    for category_name, category_url in all_categories_dict.items():
        print(f"Обработка категории: {category_name}")
        # Извлекаем субкатегории
        subcategories = parse_category_page(category_url)
        if not subcategories:
            print(f"Субкатегории не найдены для категории {category_name}")
            continue

        # Открываем CSV-файл для записи данных по данной категории
        with open(f"{category_name}.csv", 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['page_url', 'image_url', 'author', 'title', 'annotation']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Проходим по всем субкатегориям
            for subcategory_name, subcategory_url in subcategories.items():
                print(f"  Обработка субкатегории: {subcategory_name}")
                # Извлекаем ссылки на книги из субкатегории
                book_links = parse_subcategory_page(subcategory_url)
                if not book_links:
                    print(f"  Книги не найдены в субкатегории {subcategory_name}")
                    continue

                # Проходим по каждой книге и извлекаем информацию
                for book_url in book_links:
                    try:
                        book_data = parse_book_page(book_url)
                        writer.writerow(book_data)
                    except:
                        print(f'Ошибка на книге {book_url}')