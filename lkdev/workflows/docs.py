# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from ..ghactions import script
from ._common import step_checkout

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
        "concurrency": {
            "group": "doc-${{github.ref}}",
            "cancel-in-progress": True,
        },
        "permissions": {
            "contents": "read",
            "pages": "write",
            "id-token": "write",
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
    return [
        step_checkout(),
        {
            "uses": "prefix-dev/setup-pixi@v0.8.1",
            "with": {
                "pixi-version": "latest",
                "activate-environment": True,
                "environments": "doc",
                "cache-write": False,
            },
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
        "if": "github.event_name == 'push' || github.event_name == 'release'",
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
        "if": "github.event_name == 'push' || github.event_name == 'release'",
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
                "name": "🌳 Fix local git branches",
                "run": script("""
                    for branch in $(git branch -r --list 'origin/version/*'); do
                        git branch -t ${branch##origin/} $branch
                    done
                    git branch -a
                """),
            },
            {
                "name": "🛸 Set up Deno",
                "uses": "denoland/setup-deno@v1",
                "with": {"deno-version": "~1.44"},
            },
            {"name": "🧛🏼 Set up Just", "uses": "extractions/setup-just@v2"},
            {"name": "Build site content", "run": "just build"},
            {"name": "Setup Pages", "uses": "actions/configure-pages@v5"},
            {
                "name": "📦 Upload artifact",
                "uses": "actions/upload-pages-artifact@v3",
                "with": {"path": "site"},
            },
            {
                "name": "🕸️ Deploy to GitHub Pages",
                "id": "deployment",
                "uses": "actions/deploy-pages@v4",
            },
        ],
    }
