import logging

# Configure logging to output to console
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("MyScript")


def risky_function():
    return 1 / 0  # This will cause a ZeroDivisionError


def main():
    print("--- Example 1: Using logger.exception (The Shortcut) ---")
    try:
        risky_function()
    except ZeroDivisionError:
        # Automatically logs at ERROR level with stack trace
        logger.exception("Failed to calculate value")

    print("\n--- Example 2: Using exc_info=True (Explicit) ---")
    try:
        risky_function()
    except ZeroDivisionError:
        # Logs at CRITICAL level (or any other level) with stack trace
        logger.critical("Critical calculation failure!", exc_info=True)

    print(
        "\n--- Example 3: Logging exception object explicitly (Python 3.5+) ---"
    )
    try:
        risky_function()
    except ZeroDivisionError as e:
        # You can pass the exception instance directly to exc_info
        logger.error("Error occurred", exc_info=e)


if __name__ == "__main__":
    main()
