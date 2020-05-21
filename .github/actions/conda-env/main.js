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
            stdio: 'inherit',
            shell: true
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
        if (process.platform != 'win32') {
            core.info('changing $CONDA ownership');
            await exec(`sudo chown -R $USER $CONDA`, {shell: true});
        } else {
            core.warning('could not create directory in Conda, this might cause a problem');
        }
    }
}

async function initialize(cfg) {
    if (cfg.py_version) {
        core.info(`creating conda environment w/ Python ${cfg.py_version}`);
        await run(conda_bin, ['create', '-qy', '-n', cfg.name, `python=${cfg.py_version}`]);
        core.info(`updating conda environment from ${cfg.file}`);
        await run(conda_bin, ['env', 'update', '-q', '-n', cfg.name, '-f', cfg.file]);
    } else {
        core.info(`creating conda environment from ${cfg.file}`);
        await run(conda_bin, ['env', 'create', '-q', '-n', cfg.name, '-f', cfg.file]);
    }
}

async function exportUnix(cfg) {
    function parseVar(s) {
        let m = s.match(/(\w+)=(.*)/);
        if (m) {
            return {
                name: m[1],
                value: m[2]
            }
        } else {
            return null;
        }
    }

    core.info('exporting UNIX environment variables');
    let before = await exec('env', {shell: '/bin/bash'});
    let after = await exec(`_c=$(${conda_bin} shell.posix activate ${cfg.name}); eval "$_c"; env`, {shell: '/bin/bash'});
    let vars = {};
    for (let line of before.stdout.split(/\r?\n/)) {
        let v = parseVar(line);
        if (v) {
            vars[v.name] = v.value;
        }
    }
    for (let line of after.stdout.split(/\r?\n/)) {
        let v = parseVar(line);
        if (v) {
            if (vars[v.name] != v.value) {
                core.info('exporting variable ' + v.name);
                core.exportVariable(v.name, v.value);
            }
        }
    }
}

async function exportWindows(cfg) {
    core.info('exporting Windows environment variables');
    let res = await execFile(conda_bin, ['shell.powershell', 'activate', cfg.name], {shell: true});
    let vars = {};
    for (let line of res.stdout.split(/\r?\n/)) {
        let m = line.match(/^\$Env:(\w+)\s*=\s*"(.*)"/);
        if (m) {
            core.exportVariable(m[1], m[2]);
        } else if (line.trim().length) {
            core.warning('unrecognized env line: ' + line);
        }
    }
}

async function main() {
    let cfg = {
        name: core.getInput('name'),
        py_version: core.getInput('python-version'),
        file: core.getInput('env-file')
    };
    await fixPerms(cfg);
    await initialize(cfg);
    if (process.platform == 'win32') {
        await exportWindows(cfg);
    } else {
        await exportUnix(cfg);
    }
}

main().catch((err) => {
    core.setFailed(`Failed with error ${err}`);
});
