# audio_file_classifier

A tool to classify an audio file as instrumental music, music with vocal, vocal only or other and store in a json file. The purpose is to discover if a file is mixable with spoken word or field recording tracks, and originated for use with 'millicent audio mirror'

This script is currently a cli tool.

TODO: expose an end point (that returns json) that files can be sent to for classification

# Install

Make sure python 3.11 is install and in your path

Setup a virtual environment
python3.11 -m venv venv
Install dependencies
pip install -r ./requirements.txt

# Usage

clasify_audio.py -fn <filename> -vt <voiced threshold ( to find singing in vocal only class, and assign as 'vocal + music' (do not mix) )>
