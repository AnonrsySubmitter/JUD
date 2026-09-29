# Adapted from S-FLM (https://github.com/jdeschena/s-flm), Apache-2.0.
# Core generator: Ali Alp (https://github.com/alicommit-malp/sudoku), MIT.
"""S-FLM Sudoku generation with NumPy-compatible output."""

from __future__ import annotations

import random
from multiprocessing import Pool

from tqdm import tqdm

from . import layout

DIFFICULTY_TO_CLUES = {"easy": 40, "medium": 35, "hard": 30}


def _is_valid(board: list[list[int]], row: int, col: int, num: int) -> bool:
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    box_r = row - row % 3
    box_c = col - col % 3
    for i in range(3):
        for j in range(3):
            if board[box_r + i][box_c + j] == num:
                return False
    return True


def _fill_grid(grid: list[list[int]], rng: random.Random) -> bool:
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                nums = list(range(1, 10))
                rng.shuffle(nums)
                for num in nums:
                    if _is_valid(grid, i, j, num):
                        grid[i][j] = num
                        if _fill_grid(grid, rng):
                            return True
                        grid[i][j] = 0
                return False
    return True


def _count_solutions(grid: list[list[int]], limit: int = 2) -> int:
    count = [0]

    def solve(candidate: list[list[int]]) -> None:
        if count[0] >= limit:
            return
        for i in range(9):
            for j in range(9):
                if candidate[i][j] == 0:
                    for num in range(1, 10):
                        if _is_valid(candidate, i, j, num):
                            candidate[i][j] = num
                            solve(candidate)
                            candidate[i][j] = 0
                            if count[0] >= limit:
                                return
                    return
        count[0] += 1

    solve([row[:] for row in grid])
    return count[0]


def _remove_cells(grid: list[list[int]], num_clues: int, rng: random.Random) -> list[list[int]]:
    cells_to_remove = 81 - num_clues
    removed = 0
    all_cells = [(row, col) for row in range(9) for col in range(9)]
    rng.shuffle(all_cells)
    for row, col in all_cells:
        if removed >= cells_to_remove:
            break
        if grid[row][col] == 0:
            continue
        backup = grid[row][col]
        grid[row][col] = 0
        if _count_solutions(grid, limit=2) == 1:
            removed += 1
        else:
            grid[row][col] = backup
    return grid


def _generate_one(args: tuple[int, int]) -> tuple[list[list[int]], list[list[int]]]:
    seed, num_clues = args
    rng = random.Random(seed)
    grid = [[0] * 9 for _ in range(9)]
    _fill_grid(grid, rng)
    solution = [row[:] for row in grid]
    puzzle = _remove_cells(grid, num_clues, rng)
    return puzzle, solution


def _generate_raw_grids(
    num_needed: int, num_clues: int, seed: int, num_workers: int
) -> tuple[list[list[list[int]]], list[list[list[int]]]]:
    all_puzzles: list[list[list[int]]] = []
    all_solutions: list[list[list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    task_seed = seed
    progress = tqdm(total=num_needed, desc="Generating sudokus")

    while len(all_puzzles) < num_needed:
        remaining = num_needed - len(all_puzzles)
        batch_size = remaining + remaining // 10 + 16
        tasks = [(task_seed + i, num_clues) for i in range(batch_size)]
        task_seed += batch_size

        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = pool.imap(_generate_one, tasks)
        else:
            results = map(_generate_one, tasks)

        for puzzle, solution in results:
            key = tuple(cell for row in solution for cell in row)
            if key in seen:
                continue
            seen.add(key)
            all_puzzles.append(puzzle)
            all_solutions.append(solution)
            progress.update(1)
            if len(all_puzzles) >= num_needed:
                break

        if pool is not None:
            pool.terminate()
            pool.join()

    progress.close()
    return all_puzzles, all_solutions


def _flatten_grid(grid: list[list[int]]) -> list[int]:
    tokens: list[int] = []
    for row in range(9):
        tokens.extend(grid[row])
        if row < 8:
            tokens.append(layout.ROW_SEPARATOR)
    return tokens


def _tokenize_grids(
    puzzles: list[list[list[int]]], solutions: list[list[list[int]]]
) -> list[list[int]]:
    return [
        [layout.BOS, *_flatten_grid(puzzle), layout.BOS, *_flatten_grid(solution)]
        for puzzle, solution in zip(puzzles, solutions, strict=True)
    ]


def generate_sudoku_dataset(
    num_train: int = 48_000,
    num_valid: int = 2_000,
    difficulty: str = "hard",
    seed: int = 42,
    num_workers: int = 1,
) -> dict[str, list[list[int]]]:
    if difficulty not in DIFFICULTY_TO_CLUES:
        raise ValueError(f"invalid difficulty: {difficulty!r}")
    puzzles, solutions = _generate_raw_grids(
        num_train + num_valid,
        DIFFICULTY_TO_CLUES[difficulty],
        seed,
        num_workers,
    )

    rng = random.Random(seed)
    indices = list(range(len(puzzles)))
    rng.shuffle(indices)
    puzzles = [puzzles[index] for index in indices]
    solutions = [solutions[index] for index in indices]
    sequences = _tokenize_grids(puzzles, solutions)
    return {"train": sequences[:num_train], "validation": sequences[num_train:]}
