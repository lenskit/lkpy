from pathlib import Path
import requests
from zipfile import ZipFile

ML_LOC = "http://files.grouplens.org/datasets/movielens/"
ML_DATASETS = {
    'ml-100k': 'ml-100k/u.data',
    'ml-1m': 'ml-1m/ratings.dat',
    'ml-10m': 'ml-10M100K/ratings.dat',
    'ml-20m': 'ml-20m/ratings.csv',
    'ml-25m': 'ml-25m/ratings.csv',
}


def fetch_ml(dir: Path, ds: str):
    zipname = f'{ds}.zip'
    zipfile = dir / zipname
    zipurl = ML_LOC + zipname

    test_file = dir / ML_DATASETS[ds]
    if test_file.exists():
        print(test_file, 'already exists')
        return

    print('downloading data set', ds)
    with zipfile.open('wb') as zf:
        res = requests.get(zipurl, stream=True)
        for block in res.iter_content(None):
            zf.write(block)

    print('unpacking data set')
    with ZipFile(zipfile, 'r') as zf:
        zf.extractall(dir)
