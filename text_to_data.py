import os
import collections
import random
import re
import requests
import hashlib
import torch

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(url, folder='../data', sha1_hash=None):
    """Download a file to folder and return the local filepath.

    Defined in :numref:`sec_utils`"""
    if not url.startswith('http'):
        # For back compatability
        url, sha1_hash = DATA_HUB[url]
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    # Download
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


class TimeMachine():
    def _download(self):
        fname = download(DATA_URL + "timemachine.txt", self.root, "090b5e7e70c295757f55df93cb0a180b9691891a")
        with open(fname) as f:
            return f.read()

    def _preprocess(self, text):
        return re.sub('[A-Za-z]+', ' ', text).lower()

    def _tokenize(self, text):
        return list(text)





