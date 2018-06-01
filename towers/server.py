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


class CommandHandler(socketserver.StreamRequestHandler):
    def handle(self):
        k = json.load(self.rfile)
        if k == "sendCash":
            COMMANDSERVERSEMAPHORE.acquire()
            v = json.dumps(list(RESULTSCASH.items()))
            self.wfile.write(bytes(v,'ascii'))
            COMMANDSERVERSEMAPHORE.release()
        else:
            plan = k["plan"]
            perturbation = k["perturbation"]
            n = k["n"]

            k = (tuple(map(tuple, plan)), perturbation)
            if k in RESULTSCASH:
                v = RESULTSCASH[k]
            else:
                COMMANDSERVERSEMAPHORE.acquire()
                v = TowerWorld().sampleStability(plan, perturbation, n)
                RESULTSCASH[k] = v
                if powerOfTen(len(RESULTSCASH)):
                    print("Tower cache reached size", len(RESULTSCASH))
                    name = "experimentOutputs/towers%d.png" % len(RESULTSCASH)
                    exportTowers(
                        list(set([_t for _t, _ in list(RESULTSCASH.keys())])), name)
                    print("Exported towers to image", name)

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
        print(" [+] Found existing tower server")
        return

    time.sleep(0.2 + random.random())
    if command_server_running():
        print(" [+] Found existing tower server")
        return

    print(" [+] Launching tower server")
    os.system("python towers/server.py")
    time.sleep(0.5)


def kill_servers():
    ps = []
    for p in psutil.process_iter(attrs=['name', 'cmdline']):
        if p.info['name'] == 'python' and 'towers/server.py' in p.info['cmdline'] and 'KILL' not in p.info['cmdline']:
            ps.append(p.pid)
    for p in ps:
        print(" [+] Killing tower server with PID %d" % p)
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
    server.serve_forever()
