# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class FahasascraperItem(scrapy.Item):
    # define the fields for your item here like:
    name = scrapy.Field()
    pass

class BookItem(scrapy.Item):
   title = scrapy.Field()
   category = scrapy.Field()
   author = scrapy.Field()
   price = scrapy.Field()
   total_sales = scrapy.Field()
   rating = scrapy.Field()
   rating_count = scrapy.Field()
   description = scrapy.Field()
   image = scrapy.Field()
   url = scrapy.Field()
