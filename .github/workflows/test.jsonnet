local PYTHONS = ['3.10', '3.11'];
local BASIC_PLATFORMS = ['ubuntu-latest', 'macos-latest', 'windows-latest'];
local ALL_PLATFORMS = BASIC_PLATFORMS + ['macos-13'];

local runs_on(options) =
  if std.objectHas(options, 'platform')
  then options.platform
  else if std.objectHas(options, 'matrix')
  then '${{matrix.platform}}'
  else 'ubuntu-latest';

local strategy(options) =
  if std.objectHas(options, 'matrix') then {
    strategy: {
      'fail-fast': false,
      matrix: options.matrix,
    },
  } else {};

local python_ver(options) =
  if std.objectHas(options, 'matrix') then
    '${{matrix.python}}'
  else
    std.get(options, 'python-ver', PYTHONS[0]);


local steps_checkout(options) = [{
  name: '🛒 Checkout',
  uses: 'actions/checkout@v4',
  with: { 'fetch-depth': 0 },
}];

local steps_setup_conda(options) = [
  {
    name: '👢 Generate Conda environment file',
    run: std.format(
      'pipx run ./utils/conda-tool.py --env -o ci-environment.yml -e %s pyproject.toml dev-requirements.txt',
      [std.get(options, 'extra', 'all')]
    ),
  },
  {
    name: '📦 Set up Conda environment',
    uses: 'mamba-org/setup-micromamba@v1',
    id: 'setup',
    with: {
      'environment-file': 'ci-environment.yml',
      'environment-name': 'lkpy',
      'cache-environment': true,
      'init-shell': 'bash',
    },
  },
];

local steps_setup_vanilla(options) = [
  {
    name: '🐍 Set up Python',
    uses: 'actions/setup-python@v5',
    id: 'pyinstall',
    with: {
      'python-version': python_ver(options),
    },
  },
  {
    name: '🕶️ Set up uv',
    run: 'pip install -U "uv>=0.1.15"',
  },
  {
    name: '📦 Set up Python dependencies',
    id: 'install-deps',
    run: std.format(
      'uv pip install --python "$PYTHON" -r %s -e .',
      [std.get(options, 'requirements-file', 'test-requirements.txt')]
    ) + if std.objectHas(options, 'pip-args') then ' ' + options['pip-args'] else '',
    env: {
      PYTHON: '${{steps.install-python.outputs.python-path}}',
      UV_EXTRA_INDEX_URL: 'https://download.pytorch.org/whl/cpu',
      UV_INDEX_STRATEGY: 'unsafe-first-match',
    },
  },
];

local steps_inspect(options) = [{
  name: '🔍 Inspect environment',
  run: std.join('\n', [
    'python -V',
    'numba -s',
  ]),
}];

local steps_test(options) = [
  {
    name: '🏃🏻‍➡️ Test LKPY',
    run: std.format(
      'python -m pytest --cov=lenskit --verbose --log-file=%s --durations=25',
      [std.get(options, 'log-file', 'test.log')]
    ) + if std.objectHas(options, 'test-args') then
      ' ' + std.join(' ', options['test-args'])
    else '',
  } + (if std.objectHas(options, 'test-env') then {
         env: options['test-env'],
       } else {}),
  {
    name: '📐 Coverage results',
    run: 'coverage xml',
  }
  {
    name: '📤 Upload test results',
    uses: 'actions/upload-artifact@v3',
    with: {
      name: options['test-name'],
      path: std.join('\n', ['test*.log', 'coverage.xml']),
    },
  },
];

local job_steps(env, options) =
  steps_checkout(options)
  + (if env == 'conda' then steps_setup_conda(options) else steps_setup_vanilla(options))
  + steps_inspect(options)
  + steps_test(options);

local conda_pipe(options) =
  {
    name: options.name,
    'runs-on': runs_on(options),
    'timeout-minutes': 30,
  }
  + strategy(options)
  + { steps: job_steps('conda', options) };

local vanilla_pipe(options) =
  {
    name: options.name,
    'runs-on': runs_on(options),
    'timeout-minutes': 30,
  } + strategy(options)
  + { steps: job_steps('conda', options) };

std.manifestYamlDoc({
  name: 'Test Suite',
  on: {
    push: {
      branches: ['main'],
    },
    pull_request: null,
  },
  defaults: {
    run: {
      shell: 'bash -el {0}',
    },
  },
  concurrency: {
    group: 'test-${{github.ref}}',
    'cancel-in-progress': true,
  },
  jobs: {
    conda: conda_pipe({
      name: 'Conda Python ${{matrix.python}} on ${{matrix.platform}}',
      matrix: {
        python: PYTHONS,
        platform: ALL_PLATFORMS,
      },
      'test-name': 'test-conda-${{matrix.platform}}-py${{matrix.python}}',
    }),
    vanilla: vanilla_pipe({
      name: 'Vanilla Python ${{matrix.python}} on ${{matrix.platform}}',
      matrix: {
        python: PYTHONS,
        platform: BASIC_PLATFORMS,
      },
      'test-name': 'test-vanilla-${{matrix.platform}}-py${{matrix.python}}',
    }),
    nojit: conda_pipe({
      name: 'Non-JIT test',
      'test-name': 'test-nojit',
      'test-env': {
        NUMBA_DISABLE_JIT: 1,
        PYTORCH_JIT: 0,
      },
    }),
    mindep: vanilla_pipe({
      name: 'Minimal dependency test',
      'test-name': 'test-mindep',
      'pip-args': '--resolution=lowest-direct',
    }),
    results: {
      name: 'Test suite reuslts',
      'runs-on': 'ubuntu-latest',
      needs: ['conda', 'vanilla', 'nojit', 'mindep'],
      steps:
        steps_checkout({}) + [
          {
            name: '📥 Download test artifacts',
            uses: 'actions/download-artifact@v3',
            with: {
              path: 'test-logs',
            },
          },
          {
            name: '📋 List log files',
            run: 'ls -lR test-logs',
          },
          {
            name: '✅ Upload coverage',
            uses: 'codecov/codecov-action@v3',
            with: {
              directory: 'test-logs/',
            },
          },
        ],
    },
  },
}, quote_keys=false)
