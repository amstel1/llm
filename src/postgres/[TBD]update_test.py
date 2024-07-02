import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger
from postgres.config import user, password, host, port, database



if __name__ == '__main__':
    connection_str = f'postgresql://{user}:{password}@{host}:{port}/{database}'
    engine = create_engine(connection_str)
    try:
        with engine.connect() as connection:
            try:
                s = """UPDATE fridge.search_queue
                            SET searched = 1
                            WHERE search_query = 'Встраиваемый холодильник AEG SCS 61800 FF';"""

                news = s.strip("\n").strip().replace('\n', '')
                connection.execute(text(news))
                connection.commit()
                logger.warning(f"Executing query: {news}")
            except Exception as e:
                logger.error(f"error -- UPDATE: {e}")
    except Exception as e:
        logger.error(f"error2 -- UPDATE2: {e}")

    connection_str = f'postgresql://{user}:{password}@{host}:{port}/{database}'
    engine = create_engine(connection_str)
    try:
        with engine.connect() as connection_str:
            df = pd.read_sql(
                sql="select * from fridge.search_queue where search_query = 'Встраиваемый холодильник AEG SCS 61800 FF'",
                con=connection_str,
            )
            logger.debug(df.T)
    except:
        pass