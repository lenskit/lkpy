import sysconfig
import re


def conda_platform():
    plat = sysconfig.get_platform()
    if re.match(r'^macosx-.*-x86_64', plat):
        return 'osx-64'
    if re.match(r'^macosx-.*-arm64', plat):
        return 'osx-arm64'
    if re.match(r'^[Ll]inux.*-x86_64', plat):
        return 'linux-64'
    if re.match(r'^[Ll]inux.*-aarch64', plat):
        return 'linux-aarch64'
    if plat == 'win-amd64':
        return 'win-64'

    raise ValueError('unrecognized platform ' + plat)


if __name__ == '__main__':
    print(conda_platform())
