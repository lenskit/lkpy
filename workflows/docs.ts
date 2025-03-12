// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University
// Copyright (C) 2023-2025 Drexel University
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

import { Workflow, WorkflowJob } from "@lenskit/typeline/github";

import { checkoutStep } from "./lib/checkout.ts";
import { script } from "./lib/script.ts";
import { condaSetup } from "./test/conda.ts";

const build: WorkflowJob = {
  name: "Build documentation",
  "runs-on": "ubuntu-latest",
  steps: [
    checkoutStep(),
    ...condaSetup("doc"),
    {
      id: "docs",
      name: "📚 Build documentation site",
      run: script("just docs"),
    },
    {
      name: "📤 Package documentation site",
      uses: "actions/upload-artifact@v4",
      with: {
        name: "lenskit-docs",
        path: "build/doc",
      },
    },
  ],
};

const archive: WorkflowJob = {
  name: "Archive documentation",
  "runs-on": "ubuntu-latest",
  needs: ["build"],
  if: "github.event_name == 'push' || github.event_name == 'release'",
  environment: "docs",
  steps: [
    checkoutStep(1),
    {
      name: "Check out doc site",
      uses: "actions/checkout@v4",
      if: "github.event_name != 'release'",
      with: {
        repository: "lenskit/lenskit-docs",
        "ssh-key": "${{ secrets.DOC_DEPLOY_KEY }}",
        path: "doc-site",
        ref: "version/latest",
      },
    },
    {
      name: "Check out doc site (stable)",
      uses: "actions/checkout@v4",
      if: "github.event_name == 'release'",
      with: {
        repository: "lenskit/lenskit-docs",
        "ssh-key": "${{ secrets.DOC_DEPLOY_KEY }}",
        path: "doc-site",
        ref: "version/stable",
      },
    },
    {
      name: "📥 Fetch documentation package",
      uses: "actions/download-artifact@v4",
      with: {
        name: "lenskit-docs",
        path: "build/doc",
      },
    },
    {
      name: "🛻 Copy documentation content",
      run: script(
        "rsync -av --delete --exclude=.git/ --exclude=.buildinfo --exclude=.doctrees build/doc/ doc-site/",
        "cd doc-site",
        "git config user.name 'LensKit Doc Bot'",
        "git config user.email 'docbot@lenskit.org'",
        "git add .",
        "git commit -m 'rebuild documentation'",
        "git push",
      ),
    },
    {
      name: "🏷️ Tag release documentation",
      if: "github.event_name == 'release'",
      run: script(
        "cd doc-site",
        "ver=$(echo ${{github.event.release.tag_name}} | sed -e 's/^v//'",
        "git checkout -b version/$ver",
        "git push origin version/$ver",
      ),
    },
  ],
};

const publish: WorkflowJob = {
  name: "Publish documentation",
  "runs-on": "ubuntu-latest",
  needs: ["archive"],
  if: "github.event_name == 'push' || github.event_name == 'release'",
  environment: {
    name: "docs",
    url: "https://lkpy.lenskit.org",
  },
  steps: [
    {
      name: "Check out doc site",
      uses: "actions/checkout@v4",
      with: {
        repository: "lenskit/lenskit-docs",
        ref: "main",
        "fetch-depth": 0,
      },
    },
    {
      name: "🌳 Fix local git branches",
      run: script(`
                for branch in $(git branch -r --list 'origin/version/*'); do
                    git branch -t \${branch##origin/} $branch
                done
                git branch -a
            `),
    },
    {
      name: "🛸 Set up Deno",
      uses: "denoland/setup-deno@v1",
      with: { "deno-version": "~1.44" },
    },
    { name: "🧛🏼 Set up Just", uses: "extractions/setup-just@v2" },
    { name: "Build site content", run: "just build" },
    { name: "Setup Pages", uses: "actions/configure-pages@v5" },
    {
      name: "🕸️ Deploy to Netlify",
      id: "deployment",
      uses: "netlify/actions/cli@master",
      with: { args: "deploy --dir=site --prod" },
      env: {
        NETLIFY_AUTH_TOKEN: "${{secrets.NETLIFY_AUTH_TOKEN}}",
        NETLIFY_SITE_ID: "${{vars.NETLIFY_SITE_ID}}",
      },
    },
  ],
};

export const workflow: Workflow = {
  name: "Documentation",
  on: {
    push: {
      branches: ["main"],
    },
    pull_request: {},
    workflow_dispatch: {},
    release: { types: ["published"] },
  },
  concurrency: {
    group: "doc-${{github.ref}}",
    "cancel-in-progress": true,
  },
  permissions: {
    contents: "read",
    pages: "write",
    "id-token": "write",
  },
  jobs: {
    build,
    archive,
    publish,
  },
};
