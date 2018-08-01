"""
Get version information for cache keys.
"""

import sys
import json
import urllib
import httplib

mc_url = 'https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh'
pd_url = 'https://api.anaconda.org/package/anaconda/pandas/files'

print >>sys.stderr, 'fetching Miniconda installer etag'
rci = httplib.HTTPSConnection('repo.continuum.io')
rci.request('HEAD', '/miniconda/Miniconda3-latest-MacOSX-x86_64.sh')
mc_res = rci.getresponse()
mc_etag = mc_res.getheader('etag')
rci.close()
print 'installer etag', mc_etag

print >>sys.stderr, 'fetching Pandas versions'
pd_res = urllib.urlopen(pd_url)
pd_info = json.load(pd_res)
pd_osx_files = [f['basename'] for f in pd_info if f['attrs']['platform'] == 'osx']
for file in pd_osx_files:
    print 'osx distfile', file
