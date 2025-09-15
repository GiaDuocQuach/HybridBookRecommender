import scrapy
import re
from scrapy.http import Request
from fahasascraper.items import BookItem

class FahasaspiderSpider(scrapy.Spider):
    name = "fahasaspider"
    allowed_domains = ["www.fahasa.com"]#, "proxy.scrapeops.io"]
    start_urls = ["https://www.fahasa.com/sach-trong-nuoc.html?order=num_orders&limit=48&p=1cd"]

    custom_settings = {
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0",
        "DOWNLOAD_DELAY": 0.5
    }

    '''def start_requests(self):
        #yield scrapy.Request(self.start_urls[0], callback=self.parse)
        for page in range(1, 888):
            url = f"https://www.fahasa.com/sach-trong-nuoc.html?order=num_orders&limit=48&p={page}"
            yield Request(url, callback=self.parse)'''

    def parse(self, response):
        books = response.css('div.ma-box-content')

        for book in books:
            book_url = book.css('h2 a').attrib['href']
            if book_url is not None:
                yield response.follow(book_url, callback=self.parse_book_page)

        next_page = response.css('li.current + li a::text').get()
        if next_page is not None:
            next_page = 'https://www.fahasa.com/sach-trong-nuoc.html?order=num_orders&limit=48&p=' + next_page
            yield response.follow(next_page, callback= self.parse)
        '''books = response.css('h2.product-name-no-ellipsis.p-name-list a')
        if not books:
            return

        for book in books:
            book_name = book.css('::attr(title)').get()
            book_link = book.css('::attr(href)').get()
            full_link = response.urljoin(book_link)
            yield Request(full_link, callback=self.parse_book_page, meta={'book_name': book_name})'''

    def parse_book_page(self, response):

        '''book_item = BookItem()

        book_item['title'] = ' '.join(''.join(response.xpath('//h1[contains(@class, "fhs_name_product_desktop")]//text()').getall()).replace('\n', ' ').replace('\t', ' ').split())
        book_item['category'] = response.xpath('//ol[@class="breadcrumb"]/li[2]/a/text()').get()
        book_item['author'] = response.xpath('//div[@class="product-view-sa-author"]/span[2]/text()').get()
        book_item['price'] = response.xpath('//div[@class="price-box"]/p[@class="special-price"]/span[2]/text()').get().split()[0]
        book_item['total_sales'] = response.xpath('//div[@class="product-view-qty-num"]/text()').get()
        book_item['rating'] = response.xpath('//div[@class="rating"]/@style').get().split(':')[1].strip().replace(';', '')
        book_item['rating_count'] = response.xpath('//table[@class="ratings-desktop"]//a/text()').re_first(r'\((\d+)')
        book_item['description'] = re.sub(r'\s+', ' ', " ".join(response.xpath("//div[@id='desc_content']//text()").getall())).strip()'''


        book_item = BookItem()

        #book_item['title'] = response.meta['book_name']
        book_item['title'] = ' '.join(' '.join(response.css('h1.fhs_name_product_desktop ::text').getall()).split())
        book_item['category'] = response.css('ol.breadcrumb li:nth-child(2) a::text').get(default='').strip()
        book_item['author'] = response.css('div.product-view-sa-author span:nth-child(2)::text').get(default='').strip()
        book_item['price'] = response.css('p.special-price span.price::text').get(default='').strip()
        book_item['total_sales'] = response.css('div.product-view-qty-num::text').get(default='').strip()
        book_item['rating'] = response.css('div.icon-star-text::text').get(default='0').strip()
        book_item['rating_count'] = response.css('p.rating-links a::text').re_first(r'\d+').strip() if response.css('p.rating-links a::text') else ''
        book_item['description'] = re.sub(r'\s+', ' ', " ".join(response.css('#desc_content::text, #desc_content *::text').getall())).strip()
        book_item['image'] = response.css('img#image::attr(data-src)').get() or response.css('img#image::attr(src)').get(default='')
        book_item['url'] = response.url

        yield book_item