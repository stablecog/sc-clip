from utils.logger import setup_logger

logger_listener = setup_logger()

from threading import Thread, Event
import signal

from dotenv import load_dotenv

from models.setup import setup
from servers.clip import run_clipapi
import logging

# Define an event to signal all threads to exit
shutdown_event = Event()

if __name__ == "__main__":
    load_dotenv()
    models_pack = setup()

    # Setup signal handler for exit
    def signal_handler(signum, frame):
        if not shutdown_event.is_set():
            logging.info("Signal received, shutting down...")
            shutdown_event.set()
            logger_listener.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # CLIP
        clipapi_thread = Thread(target=lambda: run_clipapi(models_pack=models_pack))
        clipapi_thread.start()
        # Put other threads here

        # Join threads
        clipapi_thread.join()
        # Join other threads here
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C gracefully. The signal handler already sets the shutdown_event.
    finally:
        # Any other cleanup in the main thread you want to perform.
        logging.info("Main thread cleanup done!")
