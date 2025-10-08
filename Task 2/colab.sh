#!/bin/sh

rm -rf /content/code
cp -rf "$(pwd)" /content/code

cd /content/code

mkdir -p /results
mount --bind "$(pwd)" /results

add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install python3.8 python3.8-distutils python3.8-venv
curl https://bootstrap.pypa.io/get-pip.py | python3.8 -

python3.8 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
