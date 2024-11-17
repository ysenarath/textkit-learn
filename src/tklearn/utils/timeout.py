import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Callable, TypeVar, Union

T = TypeVar("T")


def timeout(seconds: Union[int, float]) -> Callable:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except TimeoutError:
                    raise TimeoutError(
                        f"Function '{func.__name__}' timed out after {seconds} seconds"
                    )

        return wrapper

    return decorator
