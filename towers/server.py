import codecs
import random
import time
import threading
import socket
import psutil
import os
import sys
import socketserver
import json

from tower_common import TowerWorld, exportTowers

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    flushEverything()

def flushEverything():
    sys.stdout.flush()
    sys.stderr.flush()


COMMANDSERVERPORT = 9494
COMMANDSERVERSEMAPHORE = None
MAXIMUMNUMBEROFCONNECTIONS = None


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


RESULTSCASH = {}


def powerOfTen(n):
    if n <= 0:
        return False
    while True:
        if n == 1:
            return True
        if n % 10 != 0:
            return False
        n = n / 10



def exportToRAM(content):
    import tempfile
    fd = tempfile.NamedTemporaryFile(mode="w", dir="/dev/shm", delete=False)
    n = fd.name

    json.dump(content,fd)

    fd.close()

    return n
    
class CommandHandler(socketserver.StreamRequestHandler):
    def handle(self):
        k = json.load(codecs.getreader('utf-8')(self.rfile))
        if k == "sendCash":
            COMMANDSERVERSEMAPHORE.acquire()
            v = list(RESULTSCASH.items())
            COMMANDSERVERSEMAPHORE.release()
            n = exportToRAM(v)
            self.wfile.write(bytes(json.dumps(n), 'ascii'))
        else:
            plan = k["plan"]
            perturbation = k["perturbation"]
            n = k["n"]

            k = (tuple(map(tuple, plan)), perturbation)
            if k in RESULTSCASH:
                v = RESULTSCASH[k]
                # eprint("(python: hit %s)"%str(k[0]))
            else:
                # eprint("(python: miss %s)"%str(k[0]))
                COMMANDSERVERSEMAPHORE.acquire()
                v = TowerWorld().sampleStability(plan, perturbation, n)
                RESULTSCASH[k] = v
                if powerOfTen(len(RESULTSCASH)):
                    eprint("Tower cache reached size", len(RESULTSCASH))
                    # name = "experimentOutputs/towers%d.png" % len(RESULTSCASH)
                    # exportTowers(
                    #     list(set([_t for _t, _ in list(RESULTSCASH.keys())])), name)
                    # eprint("Exported towers to image", name)

                COMMANDSERVERSEMAPHORE.release()

            v = bytes(json.dumps(v), 'ascii')
            self.wfile.write(v)


def command_server_running():
    for p in psutil.process_iter(attrs=['name', 'cmdline']):
        if p.info['name'] == 'python' and 'server.py' in p.info['cmdline'] and 'KILL' not in p.info['cmdline']:
            return True
    return False


def start_server():
    if command_server_running():
        eprint(" [+] Found existing tower server")
        return

    time.sleep(0.2 + random.random())
    if command_server_running():
        eprint(" [+] Found existing tower server")
        return

    eprint(" [+] Launching tower server")
    os.system("python towers/server.py")
    time.sleep(0.5)


def kill_servers():
    ps = []
    for p in psutil.process_iter(attrs=['name', 'cmdline']):
        if p.info['name'] == 'python' and 'towers/server.py' in p.info['cmdline'] and 'KILL' not in p.info['cmdline']:
            ps.append(p.pid)
    for p in ps:
        eprint(" [+] Killing tower server with PID %d" % p)
        os.system("kill -9 %s" % p)


def send_to_tower_server(k):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect to server and send data
        sock.connect(("localhost", COMMANDSERVERPORT))
        sock.sendall(json.dumps(k) + "\n")
        # Receive data from the server and shut down
        received = sock.recv(1024)
    finally:
        sock.close()

    return json.loads(received)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "KILL":
        kill_servers()
        sys.exit(0)

    host = "localhost"
    COMMANDSERVERSEMAPHORE = threading.Semaphore(1)

    server = ThreadedTCPServer((host, COMMANDSERVERPORT), CommandHandler)
    eprint(" [+] Binding python%s tower server on %s port %d"%(sys.version_info[0], host, COMMANDSERVERPORT))
    server.serve_forever()
