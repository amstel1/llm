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

CREATE TABLE scraped_data.product_query_attempts
(
    attempt_product_name text,
    attempt_datetime timestamp without time zone[]
);

CREATE OR REPLACE VIEW scraped_data.washing_machine
    AS
     SELECT details.brand,
    details.name,
    details.min_price AS price,
    reviews.product_rating_value AS rating_value,
    reviews.product_rating_count AS rating_count,
    reviews.product_review_count AS review_count,
    details.max_load,
    details.depth,
    details.drying
   FROM scraped_data.item_details_washing_machine details
   LEFT JOIN scraped_data.reviews_product_details reviews ON details.name = reviews.query_item_name;