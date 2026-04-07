import os
from loguru import logger
import atexit
from datetime import datetime


class LoggerSingleton:
    _instance = None
    MAX_LOG_FILES_TO_KEEP = 5

    def __new__(cls):
        if cls._instance is None:
            os.makedirs("logs", exist_ok=True)

            cls._cleanup_old_logs(cls.MAX_LOG_FILES_TO_KEEP)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"logs/session_{timestamp}.log"

            cls._instance = super(LoggerSingleton, cls).__new__(cls)
            cls._instance.logger = logger
            cls._instance.logger.remove()
            cls._instance.logger.add(log_file, level="DEBUG", enqueue=True)
            cls._instance.logger.add(lambda msg: print(msg, end=""), level="INFO")

            atexit.register(cls._instance._on_exit)

        return cls._instance

    @staticmethod
    def _cleanup_old_logs(max_to_keep):
        log_dir = "logs"
        log_files = [
            f
            for f in os.listdir(log_dir)
            if f.startswith("session_") and f.endswith(".log")
        ]
        log_files = sorted(log_files, reverse=True)

        for old_file in log_files[max_to_keep:]:
            try:
                os.remove(os.path.join(log_dir, old_file))
            except Exception as e:
                print(f"Failed to delete {old_file}: {e}")

    def _on_exit(self):
        self.logger.info("Program execution completed. Logs saved.")