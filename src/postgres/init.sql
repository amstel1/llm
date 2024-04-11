-- run on creation

-- Database: scraperdb
-- DROP DATABASE IF EXISTS scraperdb;
CREATE DATABASE scraperdb
    WITH
    OWNER = scraperuser
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8'
    LOCALE_PROVIDER = 'libc'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1
    IS_TEMPLATE = False;


-- SCHEMA: scraped_data
-- DROP SCHEMA IF EXISTS scraped_data ;
CREATE SCHEMA IF NOT EXISTS scraped_data
    AUTHORIZATION scraperuser;


-- Table: scraped_data.product_item_list
-- DROP TABLE IF EXISTS scraped_data.product_item_list;
CREATE TABLE IF NOT EXISTS scraped_data.product_item_list
(
    product_type_name text COLLATE pg_catalog."default",
    product_type_url text COLLATE pg_catalog."default",
    product_url text COLLATE pg_catalog."default",
    product_name text COLLATE pg_catalog."default",
    product_position integer,
    crawl_id integer
)
TABLESPACE pg_default;
ALTER TABLE IF EXISTS scraped_data.product_item_list
    OWNER to scraperuser;