import pandas as pd
import requests
import time
import random
from tqdm import tqdm

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
    'Referer': 'https://tiki.vn/chat-gpt-thuc-chien-p275702538.html?itm_campaign=CTP_YPD_TKA_PLA_UNK_ALL_UNK_UNK_UNK_UNK_X.298704_Y.1881024_Z.3981296_CN.PA---Chat-GPT-Thuc-Chien&itm_medium=CPC&itm_source=tiki-ads&spid=275702540',
    'X-Guest-Token': 'ujRwScf25dXrlvNCPEpHVZkQADzOI7YU',
    'Connection': 'keep-alive',
    'TE': 'Trailers',
}

params = (
    ('platform', 'web'),
    ('spid', '275702540'),
    ('version', '3')
)


def parser_product(json):
    d = dict()
    try:
        d['title'] = json.get('name', '')
        d['price'] = int(json.get('price', 0))
        d['total_sales'] = int(json.get('all_time_quantity_sold', 0))
        d['rating'] = float(json.get('rating_average', 0))
        d['rating_count'] = int(json.get('review_count', 0))
        d['author'] = ', '.join(author.get('name', '') for author in json.get('authors', [])) or ''
        d['category'] = json.get('breadcrumbs', [])[2].get('name', '')
        d['description'] = json.get('description', '')
        d['image'] = json.get('images', [])[0].get('base_url', '')
        d['short_url'] = json.get('short_url', '')
    except:
        d['title'] = ''
        d['price'] = ''
        d['rating'] = ''
        d['rating_count'] = ''
        d['total_sales'] = ''
        d['author'] = ''
        d['category'] = ''
        d['description'] = ''
        d['image'] = ''
        d['short_url'] = ''

    return d


df_id = pd.read_csv('tiki_books_id.csv')
p_ids = df_id.id.to_list()
print(p_ids)
result = []
for pid in tqdm(p_ids, total=len(p_ids)):
    try:
        response = requests.get('https://tiki.vn/api/v2/products/{}'.format(pid), headers=headers, params=params)
        if response.status_code == 200:
            print('Crawl data {} success !!!'.format(pid))
            result.append(parser_product(response.json()))
        time.sleep(random.randrange(3, 5))
    except:
        pass
df_product = pd.DataFrame(result)
df_product.to_csv('Tikidata.csv', index=False)