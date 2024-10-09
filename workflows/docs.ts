import { Workflow, WorkflowJob } from "@lenskit/typeline/github";

import { checkoutStep } from "./helpers/checkout.ts";
import { script } from "./helpers/script.ts";

const build: WorkflowJob = {
    name: "Build documentation",
    "runs-on": "ubuntu-latest",
    steps: [],
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
            with: {
                repository: "lenskit/lenskit-docs",
                "ssh-key": "${{ secrets.DOC_DEPLOY_KEY }}",
                path: "doc-site",
                ref: "version/latest",
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
            ),
        },
        {
            name: "🛫 Push documentation",
            run: "cd doc-site && git push",
        },
    ],
};
const publish: WorkflowJob = {
    name: "Publish documentation",
    "runs-on": "ubuntu-latest",
    needs: ["archive"],
    if: "github.event_name == 'push' || github.event_name == 'release'",
    environment: {
        name: "github-pages",
        url: "${{ steps.deployment.outputs.page_url }}",
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
            name: "📦 Upload artifact",
            uses: "actions/upload-pages-artifact@v3",
            with: { path: "site" },
        },
        {
            name: "🕸️ Deploy to GitHub Pages",
            id: "deployment",
            uses: "actions/deploy-pages@v4",
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
