-- run on creation
-- Create the scraperuser and grant all permissions
CREATE USER scraperuser WITH PASSWORD 'scraperpassword';
GRANT ALL PRIVILEGES ON DATABASE scraperdb TO scraperuser;

CREATE SCHEMA scraped_data;
GRANT ALL ON SCHEMA scraped_data TO scraperuser;

-- Table: scraped_data.product_item_list
DROP TABLE IF EXISTS scraped_data.product_item_list;
CREATE TABLE IF NOT EXISTS scraped_data.product_item_list
(
    product_type_name text COLLATE pg_catalog."default",
    product_type_url text COLLATE pg_catalog."default",
    product_url text COLLATE pg_catalog."default",
    product_name text COLLATE pg_catalog."default",
    product_position integer,
    scraped_datetime TIMESTAMP
)
