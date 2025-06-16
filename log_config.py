import logging
from logging.handlers import TimedRotatingFileHandler
import os
from dotenv import load_dotenv

load_dotenv()
LOG_LEVEL = os.getenv('LOG_LEVEL')

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel({
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }[LOG_LEVEL])

    handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        when="midnight",
        interval=1,
        backupCount=7,
        encoding='utf-8'
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    if not logger.handlers:  # чтобы не дублировать случайно логгер
        logger.addHandler(handler)
