import logging
import os
from datetime import datetime

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_filename = datetime.now().strftime("api_status_%Y%m%d.log")
logging.basicConfig(
    filename=os.path.join(LOG_DIR, log_filename),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("TerminalLogger")

def log_api_status(endpoint, status_code, message=""):
    """
    Log API requests, status codes, and potential errors.
    Useful for monitoring monthly limits and 429 Too Many Requests errors.
    """
    if status_code >= 400:
        logger.warning(f"API ERROR: {endpoint} | Status: {status_code} | Msg: {message}")
    else:
        logger.info(f"API SUCCESS: {endpoint} | Status: {status_code} | Msg: {message}")
