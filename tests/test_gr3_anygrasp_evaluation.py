from types import SimpleNamespace

from experiments.gr3.evaluate_anygrasp import _stratified_indices


def _dataset(task_counts: dict[str, int]) -> SimpleNamespace:
    samples = [
        SimpleNamespace(task_id=task_id)
        for task_id, count in task_counts.items()
        for _ in range(count)
    ]
    return SimpleNamespace(samples=samples)


def test_stratified_indices_cover_tasks_before_repeating():
    dataset = _dataset({"task-a": 5, "task-b": 5, "task-c": 5})

    indices, counts = _stratified_indices(dataset, max_samples=5, seed=7)

    assert len(indices) == len(set(indices)) == 5
    assert sorted(counts.values()) == [1, 2, 2]


def test_stratified_indices_are_seeded_and_stable():
    dataset = _dataset({"task-a": 5, "task-b": 5, "task-c": 5})

    first = _stratified_indices(dataset, max_samples=9, seed=11)
    repeated = _stratified_indices(dataset, max_samples=9, seed=11)
    different_seed = _stratified_indices(dataset, max_samples=9, seed=12)

    assert first == repeated
    assert first[0] != different_seed[0]
    assert first[1] == {"task-a": 3, "task-b": 3, "task-c": 3}
