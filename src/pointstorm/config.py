import logging

abstract_logger = logging.getLogger("pointstorm_log")
abstract_logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
console_handler.setFormatter(formatter)

abstract_logger.addHandler(console_handler)