"""
Load all power consumption.

Usage:
    all-power.py [<dir>]

Options:
    <dir>:
        The directory containing test results [default: test-logs]
"""

from pathlib import Path
import pandas as pd
from docopt import docopt


def main(opts):
    dir = opts['<dir>']
    if not dir:
        dir = 'test-logs'
    dir = Path(dir)
    print('scanning directory', dir)

    res = []
    for f in dir.glob('log-*/emissions.csv'):
        print('found emissions in', f.parent.name)
        key = f.parent.name[4:]
        data = pd.read_csv(f)
        data['key'] = key
        res.append(data)

    df = pd.concat(res, ignore_index=True)
    print('total power usage: {:.4f} kWh'.format(df['energy_consumed'].sum()))
    print('total emissions: {:.4f} kgCO2'.format(df['emissions'].sum()))
    df.to_csv(dir / 'emissions.csv', index=False)


if __name__ == '__main__':
    opts = docopt(__doc__)
    main(opts)
