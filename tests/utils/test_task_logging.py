from lenskit import batch
from lenskit.als import BiasedMFScorer
from lenskit.data import Dataset
from lenskit.logging.tasks import Task, TaskStatus
from lenskit.pipeline import topn_pipeline


def test_train_task(ml_ds: Dataset):
    info = BiasedMFScorer(features=50, epochs=5)
    pipe = topn_pipeline(info)

    with Task("train ImplicitMF", reset_hwm=True) as task:
        pipe.train(ml_ds)

    print(task.model_dump_json(indent=2))

    assert task.status == TaskStatus.FINISHED
    assert task.cpu_time is not None and task.cpu_time > 0
    if task.peak_memory is not None:
        assert task.peak_memory > 0

    users = ml_ds.users.ids()[:500]
    with Task("recommend", reset_hwm=True) as task:
        _recs = batch.recommend(pipe, users, n=20, n_jobs=2)  # type: ignore

    print(task.model_dump_json(indent=2))

    assert task.status == TaskStatus.FINISHED
    assert len(task.subtasks) == 2
    assert all(st.subprocess for st in task.subtasks)
