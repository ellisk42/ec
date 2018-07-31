import socket
import _thread
import sys
import os

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


def handle_client(connection):
    try:
        eprint("-> Client connected")
        while True:

            l1 = int.from_bytes(connection.recv(4), byteorder='big')
            data = connection.recv(l1)
            idRef = data.decode("utf8")

            l2 = int.from_bytes(connection.recv(4), byteorder='big')
            img = connection.recv(l2)

            if idRef != "DONE":
                score = compute_score(idRef, img)
                loss = str(score['loss'][0][0]).encode("utf8")
                connection.sendall(len(loss).to_bytes(4, byteorder='big'))
                connection.sendall(loss)
            else:
                break

    finally:
        # Clean up the connection
        connection.close()

        eprint("<- Client disconnected")


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
        c, _ = sock.accept()
        _thread.start_new_thread(handle_client, (c,))
