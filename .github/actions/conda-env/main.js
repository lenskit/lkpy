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

function run(cmd, args) {
    return new Promise((ok, fail) => {
        let p = cp.spawn(cmd, args, {
            stdio: 'inherit'
        });
        p.on('error', fail);
        p.on('exit', (code) => {
            if (code == 0) {
                ok();
            } else {
                fail("exited with code " + code);
            }
        });
    });
}

async function fixPerms(cfg) {
    try {
        await writeFile(path.join(conda_dir, 'envs', '.test-path'));
    } catch (e) {
        core.info('changing $CONDA ownership');
        await exec(`sudo chown -R $USER $CONDA`, {shell: true});
    }
}

async function initialize(cfg) {
    await run(conda_bin, ['env', 'create', '-q', '-n', cfg.name, '-f', cfg.file]);
}

function parseVar(s) {
    let m = /(\w+)=(.*)/.match(s);
    return {
        name: m.group(1),
        value: m.group(2)
    }
}

async function exportUnix(cfg) {
    let before = await exec('env', {shell: '/bin/bash'});
    let after = await exec(`_c=$(${conda_bin} shell.posix activate ${cfg.name}); eval "$_c"; env`, {shell: '/bin/bash'});
    let vars = {};
    for (let line of before.stdout.split(/\r?\n/)) {
        let {name, value} = parseVar(line);
        vars[name] = value;
    }
    for (let line of after.stdout.split(/\r?\n/)) {
        let {name, value} = parseVar(line);
        if (vars[name] != value) {
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
