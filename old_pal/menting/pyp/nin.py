import  urllib.request
urllib.request.urlretrieve(url, filename)

import wget
wget.download(url, filename)

import requests
res = requests.get(url)
with open(filename, 'rb') as f:
    f.write(res.read())