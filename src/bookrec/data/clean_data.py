import pandas as pd
import re
import argparse
import csv
from bs4 import BeautifulSoup
from unidecode import unidecode
from collections import Counter
from pyvi import ViTokenizer

VN_STOPWORDS = {
    'và','của','là','có','cho','những','trong','với','một','cái','các',
    'được','nhiều','có thể','không','cũng','đã','đang','sẽ','để','từ',
    'nên','vì','nếu','khi','mà','thì','làm','nói','biết','thấy','cần',
    'muốn','phải','quyển','cuốn','sách','tập','phần','chương','mục','đề',
    'nội dung','giới thiệu','tác giả','nhà xuất bản','năm xuất bản',
    'số trang','khổ','bìa','mềm','cứng','đóng gói','vận chuyển','giao hàng',
    'thanh toán','giảm giá','khuyến mãi','tặng kèm','miễn phí','freeship',
    'ship','cod'
}

PRICE_RANGES = {
    '<100k':     (0,       100_000),
    '100-200k':  (100_000, 200_000),
    '200-500k':  (200_000, 500_000),
    '>500k': (500_000, float('inf'))
}

POPULARITY_LEVELS = {
    'Trending': 0.8,
    'Hot':      0.6,
    'Warm':     0.4,
    'Cold':     0.0
}

CATEGORY_MAPPING = {
    'manga - comic': 'truyện tranh',
    'truyện tranh, manga, comic': 'truyện tranh',

    'thể dục thể thao - giải trí': 'thể dục - thể thao',
    'thể dục - thể thao': 'thể dục - thể thao',

    'thiếu nhi': 'thiếu nhi',
    'sách thiếu nhi': 'thiếu nhi',

    'báo - tạp trí': 'báo - tạp chí',
    'tạp chí - catalogue': 'báo - tạp chí',

    'giáo khoa - tham khảo': 'sách giáo khoa - tài liệu',
    'giáo khoa - giáo trình': 'sách giáo khoa - tài liệu',
    'giáo trình': 'sách giáo khoa - tài liệu',
    'sách giáo khoa - giáo trình': 'sách giáo khoa - tài liệu',
    'sách tham khảo': 'sách giáo khoa - tài liệu',
    'education - teaching': 'sách giáo khoa - tài liệu',
    'từ điển': 'sách giáo khoa - tài liệu',

    'văn học': 'văn học',
    'sách văn học': 'văn học',
    'fiction - literature': 'văn học',
    'tiểu sử hồi ký': 'văn học',

    'tâm lý - kỹ năng sống': 'kỹ năng sống',
    'sách kỹ năng sống': 'kỹ năng sống',
    'sách tâm lý - giới tính': 'kỹ năng sống',
    'sách kiến thức tổng hợp': 'kỹ năng sống',

    'sách học ngoại ngữ': 'ngoại ngữ',

    'chính trị - pháp lý - triết học': 'chính trị - pháp luật',
    'sách chính trị - pháp lý': 'chính trị - pháp luật',

    'kinh tế': 'kinh tế',
    'sách kinh tế': 'kinh tế',

    'lịch sử - địa lý - tôn giáo': 'lịch sử - địa lý - tôn giáo',
    'sách lịch sử': 'lịch sử - địa lý - tôn giáo',
    'sách tôn giáo - tâm linh': 'lịch sử - địa lý - tôn giáo',

    'khoa học kỹ thuật': 'khoa học - kỹ thuật',
    'sách khoa học - kỹ thuật': 'khoa học - kỹ thuật',

    'phong thủy - kinh dịch': 'phong thủy',

    'nuôi dạy con': 'gia đình',
    'nữ công gia chánh': 'gia đình',
    'sách thường thức - gia đình': 'gia đình',
    'làm vườn - thú nuôi': 'gia đình',

    'văn hóa - nghệ thuật - du lịch': 'văn hóa - nghệ thuật - du lịch',
    'sách văn hóa - địa lý - du lịch': 'văn hóa - nghệ thuật - du lịch',

    'âm nhạc - mỹ thuật - thời trang': 'nghệ thuật - thời trang',
    'điện ảnh - nhạc - họa': 'nghệ thuật - thời trang',

    'sách y học': 'y học - sức khỏe',

}

def clean_html(txt):
    if not isinstance(txt, str):
        return ''
    soup = BeautifulSoup(txt, 'html.parser')
    for tag in soup(['script','style']):
        tag.decompose()
    t = soup.get_text(separator=' ', strip=True)
    return re.sub(r'\s+', ' ', t)

def normalize_text(txt):
    t = unidecode(str(txt)).lower()
    t = re.sub(r'[^a-z0-9\s\-–\[\]\(\)]', ' ', t)
    return ' '.join(t.split())

def tokenize_vietnamese(txt):
    toks = ViTokenizer.tokenize(normalize_text(txt)).split()
    return ' '.join(toks)

def extract_key_phrases(txt, min_len=3, max_len=5):
    words = txt.split()
    if len(words) < min_len:
        return []
    phrases = []
    for i in range(len(words)-min_len+1):
        for L in range(min_len, min(max_len+1, len(words)-i+1)):
            phrases.append(' '.join(words[i:i+L]))
    freq = Counter(phrases)
    return [ph for ph,_ in freq.most_common()][:5]

def clean_sales(s):
    t = str(s).lower().strip()
    if 'k+' in t:
        return int(float(t.replace('k+','')) * 1000)
    if 'k' in t:
        return int(float(t.replace('k','')) * 1000)
    t = re.sub(r'[^\d]', '', t)
    return int(t or 0)

def clean_price(p):
    t = re.sub(r'[^\d\.,]', '', str(p))
    if ',' in t and '.' in t:
        t = t.replace('.','').replace(',','.')
    else:
        t = t.replace(',', '.')
    try:
        v = float(t)
        return int(v*1000) if v < 1000 else int(v)
    except:
        return 0

def clean_rating(r):
    t = re.sub(r'[^\d\.,]', '', str(r)).replace(',', '.')
    try:
        v = float(t)
        if v > 5:
            v = v/20 if v > 10 else v/2
        return max(0, min(5, v))
    except:
        return 0.0

def clean_rating_count(rc):
    t = re.sub(r'[^\d]', '', str(rc))
    return int(t or 0)

def calculate_trend_score(sales, rating, rc):
    ns = min(sales/1000, 1) if sales > 0 else 0
    nr = rating / 5
    nc = min(rc/100, 1) if rc > 0 else 0
    return ns*0.4 + nr*0.3 + nc*0.3

def get_price_range(p):
    for name, (mn, mx) in PRICE_RANGES.items():
        if mn <= p < mx:
            return name
    return 'unknown'

def get_popularity_level(ts):
    if pd.isna(ts) or ts == 0:
        return 'Cold'
    for lvl, th in POPULARITY_LEVELS.items():
        if ts >= th:
            return lvl
    return 'Cold'

def clean_title_field(t):
    if not isinstance(t, str):
        return ''
    tmp = re.sub(r"\([^)]*\)", "", t)     # bỏ ()
    tmp = re.sub(r"\[[^]]*\]", "", tmp)   # bỏ []
    head = tmp.split('–',1)[0].split('-',1)[0]
    return head.strip()

def process_fahasa(df):
    df['price']        = df['price'].apply(clean_price)
    df['total_sales']  = df['total_sales'].apply(clean_sales)
    df['rating']       = df['rating'].apply(clean_rating)
    df['rating_count'] = df['rating_count'].apply(clean_rating_count)
    return df

def process_tiki(df):
    if 'name' in df.columns:
        df.rename(columns={'name':'title'}, inplace=True)
    if 'short_description' in df.columns:
        df.rename(columns={'short_description':'description'}, inplace=True)
    elif 'description_html' in df.columns:
        df.rename(columns={'description_html':'description'}, inplace=True)
    df.rename(columns={'rating_average':'rating','review_count':'rating_count'}, inplace=True)
    df['price']        = df['price'].apply(clean_price)
    df['rating']       = df['rating'].apply(clean_rating)
    df['rating_count'] = df['rating_count'].apply(clean_rating_count)
    df['total_sales']  = df.get('total_sales', 0).apply(clean_sales) if 'total_sales' in df.columns else 0
    return df

def clean_single(path, source):
    df = pd.read_csv(path,
        sep=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
        engine='python', encoding='utf-8', on_bad_lines='skip'
    ).fillna('').replace(r'(?i)^\s*Not Found\s*$', '', regex=True)

    df['image'] = df.get('image','')
    df['url']   = df.get('url','')

    df = process_fahasa(df) if source=='fahasa' else process_tiki(df)

    valid = (
        df['title'].astype(bool).astype(int)
      + df['description'].astype(bool).astype(int)
      + df['author'].astype(bool).astype(int)
      + (df['price']>0).astype(int)
    ) >= 2
    df = df.loc[valid].copy()

    key = df['title'].str.lower() + '_' + df['author'].str.lower()
    df = df.loc[~key.duplicated()].copy()

    df['category'] = (df['category'].str.strip().str.lower().map(CATEGORY_MAPPING).fillna(df['category'].str.strip().str.lower()))
    df['author']          = df['author'].str.strip()
    df['clean_title']     = df['title'].apply(clean_title_field)
    df['description']     = df['description'].apply(clean_html)
    df['clean_desc']      = df['description'].apply(tokenize_vietnamese)
    df['clean_desc']      = df['clean_desc'].mask(df['clean_desc']=='', df['clean_title'])
    df['desc_word_count'] = df['clean_desc'].str.split().str.len()
    df['key_phrases']     = df['clean_desc'].apply(extract_key_phrases)
    df['price_k']         = (df['price']/1000).round(1)
    df['trust_score']     = df['rating']*df['rating_count']/(df['rating_count']+1)
    df['trend_score']     = df.apply(lambda x: calculate_trend_score(
                               x['total_sales'],x['rating'],x['rating_count']), axis=1)
    df['value_score']     = df['rating']/(df['price']/100000).replace({0:1})
    df['price_range']     = df['price'].apply(get_price_range)
    df['popularity_level']= df['trend_score'].apply(get_popularity_level)

    cols = [
        'title','clean_title','author','category',
        'price','price_k','price_range',
        'rating','rating_count','total_sales',
        'trust_score','trend_score','value_score','popularity_level',
        'description','clean_desc','desc_word_count','key_phrases',
        'image','url'
    ]
    return df[cols]

def clean_data(inputs, sources, output_file):
    dfs = [clean_single(p, s) for p,s in zip(inputs, sources)]
    all_df = pd.concat(dfs, ignore_index=True)
    key2 = all_df['title'].str.lower() + '_' + all_df['author'].str.lower()
    all_df = all_df.loc[~key2.duplicated()].copy()
    all_df.to_csv(output_file, index=False, encoding='utf-8')
    return all_df

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['fahasa','tiki','all'], required=True)
    parser.add_argument('--input',  nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    if args.source in ['fahasa','tiki']:
        out = clean_single(args.input[0], args.source)
    else:
        out = clean_data(args.input, ['fahasa','tiki'], args.output)

    out.to_csv(args.output, index=False, encoding='utf-8')
    print("Đã lưu vào:", args.output)
    print(out.info())
    print("\nPrice ranges:\n", out['price_range'].value_counts())
    print("\nPopularity level:\n", out['popularity_level'].value_counts())