const fs = require('fs');
const core = require('@actions/core');
const cp = require('child_process');
const util = require('util');
const path = require('path');
const exec = util.promisify(cp.exec);
const execFile = util.promisify(cp.execFile);
const writeFile = util.promisify(fs.writeFile);

const conda_dir = process.env.CONDA;
const conda_bin = path.join(conda_dir, 'condabin', 'conda');

async function fixPerms(cfg) {
    try {
        await writeFile(path.join(conda_dir, 'envs', '.test-path'));
    } catch (e) {
        core.info('changing $CONDA ownership');
        await exec(`sudo chown -R $USER $CONDA`, {shell: true});
    }
}

async function initialize(cfg) {
    await execFile(conda_bin, ['env', 'create', '-q', '-n', cfg.name, '-f', cfg.file]);
}

async function exportUnix(cfg) {
    let before = await exec('env', {shell: true});
    let after = await exec(`_c=\`${conda_bin} shell.posix activate ${cfg.name}\`; eval "$_c"; env`, {shell: true})
    let vars = {};
    for (let line of before.stdout.split(/\r?\n/)) {
        let [name, val] = line.split(/=/, 2);
        vars[name] = val;
    }
    for (let line of after.stdout.split(/\r?\n/)) {
        let [name, val] = line.split(/=/, 2);
        if (vars[name] != val) {
            core.info('exporting variable ' + name);
            core.exportVariable(name, val);
        }
    }
}

async function main() {
    let cfg = {
        name: core.getInput('name'),
        file: core.getInput('env-file')
    };
    await fixPerms(cfg);
    await initialize(cfg);
    await exportUnix(cfg);
}

main().catch((err) => {
    core.setFailed(`Failed with error ${err}`);
});
