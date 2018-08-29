from python_server import *
from utilities import *

from multiprocessing import Pool
import cache_pb2
import codecs
import random
import time
import threading
import psutil
import os
import sys
import json

from tower_common import TowerWorld, exportTowers

COMMANDSERVERPORT = 9494
CASHSEMAPHORE = None
SERIALIZEDSEMAPHORE = None
WORKERSEMAPHORE = None
MAXIMUMNUMBEROFCONNECTIONS = None
WORKERS = None




RESULTSCASH = {}


SERIALIZEDCASH = cache_pb2.TowerCash()
LASTSERIALIZED = None
def addToSerialized(plan, result):
    global LASTSERIALIZED
    LASTSERIALIZED = None
    entry = SERIALIZEDCASH.entries.add()
    entry.height = result['height']
    entry.stability = result['stability']
    entry.area = result['area']
    entry.length = result['length']
    entry.overpass = result['overpass']
    entry.staircase = result['staircase']

    for x,w,h in plan:
        b = entry.plan.add()
        b.w10 = int(w*10 + 0.5)
        b.h10 = int(h*10 + 0.5)
        b.x10 = int(x*10 + 0.5)
def outputSerialized():
    global LASTSERIALIZED
    if LASTSERIALIZED is not None: return LASTSERIALIZED
    
    import tempfile
    fd = tempfile.NamedTemporaryFile(mode="wb", dir="/dev/shm", delete=False)
    n = fd.name
    fd.write(SERIALIZEDCASH.SerializeToString())
    fd.close()
    LASTSERIALIZED = n
    return n

def inputSerialized(n):
    with open(n,'rb') as handle:
        c = cache_pb2.TowerCash()
        stuff = handle.read()
        c.ParseFromString(stuff)
        print(c)
    return c
    



def exportToRAM(content):
    import tempfile
    fd = tempfile.NamedTemporaryFile(mode="w", dir="/dev/shm", delete=False)
    n = fd.name

    json.dump(content,fd)

    fd.close()

    return n

def runSimulation(plan, perturbation, n):
    try:
        v = TowerWorld().sampleStability(plan, perturbation, n)
    except:
        v = TowerWorld.BADRESULT
    return v


def handleTowerRequest(request):
    k = json.loads(request.decode("utf-8"))
    if k == "doNothing":
        response = "noop"
    elif k == "sendSerializedCash":
        SERIALIZEDSEMAPHORE.acquire()
        n = outputSerialized()
        SERIALIZEDSEMAPHORE.release()
        response = n
    else:
        plan = k["plan"]
        perturbation = k["perturbation"]
        n = k["n"]

        k = (tuple(map(tuple, plan)), perturbation)
        CASHSEMAPHORE.acquire()
        if k in RESULTSCASH:
            v = RESULTSCASH[k]
            CASHSEMAPHORE.release()
        else:
            CASHSEMAPHORE.release()

            WORKERSEMAPHORE.acquire()
            v = WORKERS.apply_async(runSimulation,
                                    args=(plan, perturbation, n))
            WORKERSEMAPHORE.release()
            v = v.get()

            CASHSEMAPHORE.acquire()
            RESULTSCASH[k] = v
            if powerOfTen(len(RESULTSCASH)):
                eprint("Tower cache reached size", len(RESULTSCASH))
            CASHSEMAPHORE.release()


            SERIALIZEDSEMAPHORE.acquire()
            addToSerialized(plan, v)
            SERIALIZEDSEMAPHORE.release()

        # eprint("Writing out simulation result")
        response = v

    return bytes(json.dumps(response),'utf-8')


def tower_server_running():
    for p in psutil.process_iter(attrs=['name', 'cmdline']):
        if p.info['name'] == 'python' and 'tower_server.py' in p.info['cmdline'] and 'KILL' not in p.info['cmdline']:
            return True
    return False


def start_tower_server():
    if tower_command_server_running():
        eprint(" [+] Found existing tower server")
        return

    time.sleep(0.2 + random.random())
    if tower_server_running():
        eprint(" [+] Found existing tower server")
        return

    eprint(" [+] Launching tower server")
    os.system("python tower_server.py")
    time.sleep(0.5)


def kill_servers():
    ps = []
    for p in psutil.process_iter(attrs=['name', 'cmdline']):
        if p.info['name'] == 'python' and 'tower_server.py' in p.info['cmdline'] and 'KILL' not in p.info['cmdline']:
            ps.append(p.pid)
    for p in ps:
        eprint(" [+] Killing tower server with PID %d" % p)
        os.system("kill -9 %s" % p)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "KILL":
        kill_servers()
        sys.exit(0)

    CASHSEMAPHORE = threading.Semaphore(1)
    SERIALIZEDSEMAPHORE = threading.Semaphore(1)
    WORKERSEMAPHORE = threading.Semaphore(1)
    nc = numberOfCPUs()
    WORKERS = Pool(nc)

    runPythonServer(COMMANDSERVERPORT, handleTowerRequest, threads=nc)
else: assert False
