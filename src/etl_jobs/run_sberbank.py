from base import Job
import sys
sys.path.append('/home/amstel/llm/src')
from web_scraping.bank_scraper.sberbank_scraper import SberbankWebsiteRead
from rag.make_summary import SberbankWebsiteSummaryDo
from rag.hybrid_rag_insert_w_summary import SberbankWebsiteSummaryWrite
from loguru import logger

class DoWriteJob(Job):
    def run(self, data=None, processed_data=None) -> None:
        """
        Execute the ETL job by reading, processing, and writing data.
        """
        if self.reader:
            logger.info('start READ')
            data = self.reader.read()  # must be Dict with key StepNum
            self.data.update(data)
            logger.info('end READ')
        if self.processor:
            logger.info('start DO')
            processed_data = self.processor.process()  # Dict[StepNum, List[Dict[str, List[Dict]]]]
            self.processed_data.update(processed_data)
            logger.info('end DO')
        if self.writer:
            logger.info('start WRITE')
            if self.processor:
                self.writer.write(data=self.processed_data)
            else:
                self.writer.write(data=self.data)
            logger.info('end WRITE')

if __name__ == '__main__':
    sberbank_scraping = DoWriteJob(
        # reader=SberbankWebsiteRead(),
        processor=SberbankWebsiteSummaryDo(),
        writer=SberbankWebsiteSummaryWrite()
    )
    sberbank_scraping.run()