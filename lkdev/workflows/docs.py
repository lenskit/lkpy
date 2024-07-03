from ..ghactions import script
from ._common import PACKAGES, step_checkout

PYTHON_VERSION = "3.11"


def workflow():
    return {
        "name": "Documentation",
        "on": {
            "push": {
                "branches": ["main"],
            },
            "pull_request": {},
            "workflow_dispatch": {},
        },
        "defaults": {
            "run": {
                "shell": "bash -el {0}",
            },
        },
        "concurrency": {
            "group": "doc-${{github.ref}}",
            "cancel-in-progress": True,
        },
        "jobs": {
            "build": job_build_docs(),
            "archive": job_archive_docs(),
            "publish": job_publish_site(),
        },
    }


def job_build_docs():
    return {
        "name": "Build documentation",
        "runs-on": "ubuntu-latest",
        "steps": stages_setup() + stages_build_doc() + stages_package(),
    }


def stages_setup():
    pip = ["pip install --no-deps"]
    pip += [f"-e {pkg}" for pkg in PACKAGES]
    return [
        step_checkout(),
        {
            "id": "setup-env",
            "name": "📦 Set up Conda environment",
            "uses": "mamba-org/setup-micromamba@v1",
            "with": {
                "environment-file": "docs/environment.yml",
                "environment-name": "lkpy",
                "init-shell": "bash",
            },
        },
        {
            "id": "install",
            "name": "🍱 Install LensKit packages",
            "run": script.command(pip),
        },
    ]


def stages_build_doc():
    return [
        {
            "id": "docs",
            "name": "📚 Build documentation site",
            "run": script.command(["just docs"]),
        }
    ]


def stages_package():
    return [
        {
            "name": "📤 Package documentation site",
            "uses": "actions/upload-artifact@v4",
            "with": {
                "name": "lenskit-docs",
                "path": "build/doc",
            },
        }
    ]


def job_archive_docs():
    return {
        "name": "Archive documentation",
        "runs-on": "ubuntu-latest",
        "needs": ["build"],
        "environment": "docs",
        "steps": [
            step_checkout(depth=1),
            {
                "name": "Check out doc site",
                "uses": "actions/checkout@v4",
                "with": {
                    "repository": "lenskit/lenskit-docs",
                    "ssh-key": "${{ secrets.DOC_DEPLOY_KEY }}",
                    "path": "doc-site",
                    "ref": "version/latest",
                },
            },
            {
                "name": "📥 Fetch documentation package",
                "uses": "actions/download-artifact@v4",
                "with": {
                    "name": "lenskit-docs",
                    "path": "build/doc",
                },
            },
            {
                "name": "🛻 Copy documentation content",
                "run": script("""
                    rsync -av --delete --exclude=.git/ --exclude=.buildinfo --exclude=.doctrees \\
                        build/doc/ doc-site/
                    cd doc-site
                    git config user.name "LensKit Doc Bot"
                    git config user.email "docbot@lenskit.org"
                    git add .
                    git commit -m 'rebuild documentation'
                """),
            },
            {
                "name": "🛫 Push documentation",
                "run": "cd doc-site && git push",
            },
        ],
    }


def job_publish_site():
    return {
        "name": "Publish documentation",
        "runs-on": "ubuntu-latest",
        "needs": ["archive"],
        "if": 'github.event_name == "push" || github.event_name == "release"',
        "environment": {"name": "github-pages", "url": "${{ steps.deployment.outputs.page_url }}"},
        "steps": [
            {
                "name": "Check out doc site",
                "uses": "actions/checkout@v4",
                "with": {
                    "repository": "lenskit/lenskit-docs",
                    "ref": "main",
                    "fetch-depth": 0,
                },
            },
            {
                "name": "Set up Deno",
                "uses": "denoland/setup-deno@v1",
                "with": {"deno-version": "~1.44"},
            },
            {"name": "Set up Just", "uses": "extractions/setup-just@v2"},
            {"name": "Build site content", "run": "just build"},
            {"name": "Setup Pages", "uses": "actions/configure-pages@v5"},
            {
                "name": "Upload artifact",
                "uses": "actions/upload-pages-artifact@v3",
                "with": {"path": "site"},
            },
            {
                "name": "Deploy to GitHub Pages",
                "id": "deployment",
                "uses": "actions/deploy-pages@v4",
            },
        ],
    }
