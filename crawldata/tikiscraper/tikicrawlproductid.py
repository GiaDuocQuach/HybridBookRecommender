import requests
import time
import random
import pandas as pd

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
    'Referer': 'https://tiki.vn/nha-sach-tiki/c8322',
    'X-Guest-Token': 'ujRwScf25dXrlvNCPEpHVZkQADzOI7YU',
    'Connection': 'keep-alive',
    'TE': 'Trailers',
}

params = {
    'limit':'40',
    'include':'advertisement',
    'aggregations':'2',
    'version':'home-persionalized',
    'trackity_id': '6e381fa4-2424-6759-a047-508b4e05acb9',
    'category':'8322',
    'page':'1',
    'src': 'c8322',
    'urlKey':'nha-sach-tiki',
}
product_id = []

for i in range(1, 257):
  params['page'] = i
  response = requests.get('https://tiki.vn/api/personalish/v1/blocks/listings', headers = headers, params = params)
  if response.status_code == 200:
    print('request success!!')
    for record in response.json().get('data'):
      product_id.append({id: record.get('id')})
  else:
    print('NOT SUCCESS!')
  time.sleep(random.randrange(3,5))

df = pd.DataFrame(product_id)
df.columns=['id']
df.to_csv('tiki_books_id.csv', index = False)