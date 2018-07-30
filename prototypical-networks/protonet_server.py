import socket
import sys
import os

import time

from protonet_score import PretrainedProtonetDistScore, \
                           load_image_path, load_image

cache = {}
model = PretrainedProtonetDistScore("./results/best_model.pt")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()


def compute_score(idRef, img):
    if (idRef, img) in cache:
        return cache[(idRef, img)]
    else:
        x = load_image_path(idRef)
        y = load_image(img)
        score = model.score(x, y)
        cache[(idRef, img)] = score
        return score


if __name__ == "__main__":

    server_address = "./protonet_socket"

    try:
        os.unlink(server_address)
    except OSError:
        if os.path.exists(server_address):
            raise

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(server_address)
    sock.listen(1)

    while True:
        eprint("Protonet server waiting for a connection")
        connection, client_address = sock.accept()
        try:
            eprint("Client connected")

            l1 = int.from_bytes(connection.recv(4), byteorder='big')
            data = connection.recv(l1)
            idRef = data.decode("utf8")

            l2 = int.from_bytes(connection.recv(4), byteorder='big')
            img = connection.recv(l2)

            score = compute_score(idRef, img)
            dist = str(score['dist'][0][0]).encode("utf8")
            connection.sendall(len(dist).to_bytes(4, byteorder='big'))
            connection.sendall(dist)

        finally:
            # Clean up the connection
            connection.close()

            eprint("Client exiting : closing")
