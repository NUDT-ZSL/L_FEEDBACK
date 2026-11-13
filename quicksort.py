import random
from typing import List, TypeVar

T = TypeVar("T")


def quicksort(items: List[T]) -> List[T]:
    """返回按升序排列的新列表，不修改原列表。"""
    result = list(items)
    _quicksort(result, 0, len(result) - 1)
    return result


def _quicksort(arr: List[T], low: int, high: int) -> None:
    while low < high:
        pivot_index = _partition(arr, low, high)

        # 先递归较短的一侧，将栈深度控制在 O(log n)。
        if pivot_index - low < high - pivot_index:
            _quicksort(arr, low, pivot_index - 1)
            low = pivot_index + 1
        else:
            _quicksort(arr, pivot_index + 1, high)
            high = pivot_index - 1


def _partition(arr: List[T], low: int, high: int) -> int:
    # 随机选择基准值，降低有序或接近有序输入退化为 O(n²) 的概率。
    pivot_index = random.randint(low, high)
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
    pivot = arr[high]

    smaller = low
    for current in range(low, high):
        if arr[current] <= pivot:
            arr[smaller], arr[current] = arr[current], arr[smaller]
            smaller += 1

    arr[smaller], arr[high] = arr[high], arr[smaller]
    return smaller


if __name__ == "__main__":
    numbers = [8, 3, 1, 7, 0, 10, 2, 3, 9]
    print(f"排序前: {numbers}")
    print(f"排序后: {quicksort(numbers)}")
