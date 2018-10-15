from utilities import *

import threading
import zmq
import time

def worker_routine(worker_url, handler, args, context=None):
    """Worker routine"""
    context = context or zmq.Context.instance()
    # Socket to talk to dispatcher
    socket = context.socket(zmq.REP)

    socket.connect(worker_url)

    while True:
        socket.send(handler(socket.recv(), *args))
        

def runPythonServer(port, handler, threads=1, args=()):
    """Server routine"""

    url_worker = "inproc://workers"
    url_client = "tcp://127.0.0.1:%s"%port

    # Prepare our context and sockets
    context = zmq.Context.instance()

    # Socket to talk to clients
    clients = context.socket(zmq.ROUTER)
    clients.bind(url_client)

    # Socket to talk to workers
    workers = context.socket(zmq.DEALER)
    workers.bind(url_worker)

    # Launch pool of worker threads
    for i in range(threads):
        thread = threading.Thread(target=worker_routine, args=(url_worker,handler,args))
        thread.start()

    eprint("Spawned %d worker threads for server bound to %s"%(threads, url_client))

    zmq.proxy(clients, workers)

    # We never get here but clean up anyhow
    clients.close()
    workers.close()
    context.term()

# DEMO
def echoHandler(s):
    eprint("Received request: [ %s ] @ %s" % (s, time.time()))
    time.sleep(1)
    return s



if __name__ == "__main__":
    runPythonServer(9119, echoHandler, threads=2)


