from pathlib import Path
import requests
from invoke import task

from lkbuild.tasks import *

BIBTEX_URL = 'https://paperpile.com/eb/YdOlWmnlit'
BIBTEX_FILE = Path('doc/lenskit.bib')


@task
def update_bibtex(c):
    "Update the BibTeX file"
    res = requests.get(BIBTEX_URL)
    print('updating file', BIBTEX_FILE)
    BIBTEX_FILE.write_text(res.text, encoding='utf-8')
