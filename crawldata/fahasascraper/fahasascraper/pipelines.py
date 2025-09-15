# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import psycopg2

class FahasascraperPipeline:
    def process_item(self, item, spider):
        return item

class SaveToPostgresPipeline:

    def __init__(self):
        ## Connection Details
        hostname = 'localhost'
        username = 'postgres'
        password = 'giaduocquach16'
        database = 'IT5021'

        ## Create/Connect to database
        self.connection = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)

        ## Create cursor, used to execute commands
        self.cur = self.connection.cursor()

        ## Create books table if none exists
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS Fahasabooks(
            id serial PRIMARY KEY,
            title text,
            category VARCHAR(255),
            author text,
            price text,
            total_sales text,
            rating text,
            rating_count INTEGER,
            description text,
            image text,
            url text
        )
        """)

    def process_item(self, item, spider):
        ## Define insert statement
        self.cur.execute(""" insert into Fahasabooks ( 
            title,
            category,
            author,
            price,
            total_sales,
            rating,
            rating_count,
            description,
            image,
            url
            ) values (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
                )""", (
            item["title"],
            item["category"],
            item["author"],
            item["price"],
            item["total_sales"],
            item["rating"],
            item["rating_count"],
            str(item["description"]),
            item["image"],
            item["url"]
        ))

        ## Execute insert of data into database
        self.connection.commit()
        return item

    def close_spider(self, spider):
        ## Close cursor & connection to database
        self.cur.close()
        self.connection.close()