from lkbuild.tasks import *


@task
def update_env_specs(c):
    "Update environment files."
    c.run("pyproject2conda project")


@task
def docs(c, watch=False, rebuild=False):
    rb = "-a" if rebuild else ""
    if watch:
        c.run(f"sphinx-autobuild {rb} --watch lenskit docs build/doc")
    else:
        c.run(f"sphinx-build {rb} docs build/doc")
