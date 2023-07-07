import os
import sys
import shutil
import time
import numpy as np
from flask import Flask, render_template, Response, send_file, send_from_directory, request, jsonify
import sys
# import librosa
# Tornado web server
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from bark.SynthesizeThread import SynthesizeThread
# Debug logger
import logging

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
synthesize_thread = SynthesizeThread()
synthesize_thread.start()
# Initialize Flask.
app = Flask(__name__)


# Route to render GUI
@app.route('/')
def show_entries():
    return render_template('index.html')


# Route to synthesize
@app.route('/synthesize', methods=["POST"])
def synthesize():
    text = request.form['text']
    print(text)
    directory_path = 'bark/static'
    shutil.rmtree(directory_path)
    os.mkdir(directory_path)
    synthesize_thread.synthesize_queue.append(text)
    time.sleep(1)
    while not os.path.exists(f'{directory_path}/audio_0.wav'):
        time.sleep(0.01)
    general_Data = {
        'title': 'Audio Player'
    }
    return render_template('audio_play.html', **general_Data)


@app.route('/audio/<path:filename>')
def serve_audio(filename):
    server_directory = 'static'
    directory_path = 'bark/static'
    while not os.path.exists(f'{directory_path}/{filename}') and synthesize_thread.isWorking:
        time.sleep(0.01)
    return send_from_directory(server_directory, filename)
    # def generate():
    #     with open(f'{directory_path}/{filename}', "rb") as fwav:
    #         data = fwav.read(1024)
    #         while data:
    #             yield data
    #             data = fwav.read(1024)
    #
    # if os.path.exists(f'{directory_path}/{filename}'):
    #     return Response(generate(), mimetype="audio/wav")
    # else:
    #     return send_from_directory('static', filename)

# launch a Tornado server with HTTPServer.
if __name__ == "__main__":
    port = 5000
    http_server = HTTPServer(WSGIContainer(app))
    logging.debug("Started Server, Kindly visit http://0.0.0.0:" + str(port))
    http_server.listen(port, address='0.0.0.0')
    IOLoop.instance().start()
    # app.run(host='0.0.0.0', port=80, debug=True)
