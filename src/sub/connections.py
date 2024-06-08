import os
import pickle
import socket
import threading
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

from .config import HEADERLENGTH

loopsigns = ["|", "/", "-", "\\"]


class NodeConnection:
    """
    Parent class used to define common interfaces for the connections.
    """

    # TODO: improve msg format
    msg_format = {"sample_index": 0, "data": None, "stop": False}
    # If set, the loop in 'run' will keep on running
    running = threading.Event()
    name = "connection"

    verb = False

    def __init__(self, **kwargs):
        if "verb" in kwargs and kwargs["verb"] is True:
            self.verb = True

    def run(self):
        pass

    def launch(self, **kwargs):
        """
        Launch thread with 'run' as target.
        """
        if self.verb:
            print(f"[INFO] Starting {self.name.replace('_', ' ')} thread")
        self.running.set()
        self.running_thread = threading.Thread(
            target=self.run, name=self.name, daemon=True, kwargs=kwargs
        )
        self.running_thread.start()

    def kill(self):
        """
        Interrupt connection and close socket.
        """
        # TODO: use signals
        # Another possible approach could be to either wait for a message in the queue
        # or for the running event to be cleared (this is what blocks the loop)
        pass


class InputNodeConnection(NodeConnection):
    """
    Class used to handle the input messages in each network node.
    The task of `InputNodeConnection` objects is to receive incoming messages (handling
    the input socket) and place them in the input message queue of the node.
    This queue is FIFO, and messages will be extracted by the main processing thread to
    be processed.

    The `run` method is supposed to run as a separate thread, allowing continuous
    operation of the processing thread (without blocking due to slow message
    transmission).

    Message are of JSON format (see `self.msg_format`).
    The execution loop will be automatically interrupted when receiving a message with
    the `"stop"` field set to True.
    """

    def __init__(
        self,
        config: Dict,
        prev_node: Dict[str, Any],
        queue: deque,
        event_callback: threading.Event,
        max_tries: int = 30,
        **kwargs,
    ):
        """
        Initialize input socket (used to receive the activations from the previous
        node).
        Note: since the input socket acts as server, it does not need to know in advance
        the previous node's properties (addr + port), but rather it will obtain them
        when the previous node will send a request to connect.

        Args:
            config: dict containing (necessarily) the keys:
                addr: IP address of the node
                inference:
                    port_in: port for the socket
            queue: input queue of the node - used to pass the messages between this
                class and GPTServer
            event_callback: event that needs to be 'set' when queue is NOT empty
            max_tries (default: 30): maximum number of tries for setting up the
                connection
            **kwargs
        """
        if not "addr" in config:
            raise ValueError("Missing IP address in configuration")
        if not "inference" in config or not "port_in" in config["inference"]:
            raise ValueError("Missing input port in configuration")

        super().__init__(**kwargs)

        self.name = "input_queue"
        self.message_queue = queue
        self.queue_not_empty = event_callback
        self.queue_not_empty.clear()
        self.config = config
        self.prev_node_info = prev_node
        self.sock_to_prev = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        failing = True
        tries = 0
        if self.verb:
            print("[INFO] Opening socket to previous node")
        while failing and tries < max_tries:
            try:
                self.sock_to_prev.bind(
                    (self.config["addr"], self.config["inference"]["port_in"])
                )
            except Exception as e:
                print(e)
                tries += 1
                if self.verb:
                    print(
                        "> Unable to bind socket address; retrying in 1s "
                        f"{loopsigns[tries % 4]}",
                        end="\r",
                    )
                time.sleep(1)
            else:
                failing = False

        if failing:
            raise ConnectionError("Unable to connect to previous node!")

        if self.verb:
            print("[INFO] Connecting to previous node                      ")

        self.sock_to_prev.listen(1)

        connected = False
        # TODO: add timeout
        while not connected:
            self.sock_to_prev_properties = self.sock_to_prev.accept()
            if self.sock_to_prev_properties[1][0] == self.prev_node_info["addr"]:
                connected = True
            else:
                self.sock_to_prev_properties[0].close()

        if self.verb and connected:
            print("> Connection with previous node established!                     ")

    def recv_msg(self, size: int) -> bytes:
        """
        Receive a message of the specified size from the previous node.

        Remark: the size specified in socket.recv(...) is the MAX size that will be read
        from the receiver buffer.

        Args:
            size: size (in bytes) of the expected message

        Returns:
            the received message (NOT decoded)
        """
        full_msg = b""
        while self.running.is_set() and len(full_msg) < size:
            msg = self.sock_to_prev_properties[0].recv(size - len(full_msg))
            if not msg:
                # Prev node shut connection down (error)
                print("[THR] Connection was terminated unexpectedly!")
                self.running.clear()
            full_msg += msg
            if not self.running.is_set():
                break
        return full_msg

    def run(self):
        """
        Receive messages and place them in the input queue.
        """
        _n_recv_msg = 0

        while self.running.is_set():
            # Receive information from the new socket (exact length)
            msg = self.recv_msg(HEADERLENGTH)

            # Extract message length from the header
            msg_len = int(msg[:HEADERLENGTH])
            _n_recv_msg += 1

            # Read payload (exact size - this is important)
            msg_payload = self.recv_msg(msg_len)
            try:
                data = pickle.loads(msg_payload)
            except EOFError:
                # Here at the end of generation
                # FIXME: maybe trigger shutdown by clearing self.running?
                pass
            else:
                # Look for stopping msg
                if "stop" in data and data["stop"] == True:
                    # Stopping sequence
                    if self.verb:
                        print("[THR] Stopping message received! Generation complete!")
                    self.message_queue.append(data)
                    self.queue_not_empty.set()
                    # FIXME - APP: shouldn't interrupt loop - stop is only for the specific sample
                    self.running.clear()
                else:  # Not here if stopping message is received
                    self.message_queue.append(data)
                    self.queue_not_empty.set()

        if self.verb:
            print("[THR] Input queue thread stopped")

    def shutdown(self):
        self.running.clear()
        self.running_thread.join()
        if self.verb:
            print("[THR] Closing input socket")
        try:
            self.sock_to_prev_properties[0].close()
        except:
            # Here if no active connection, only bound socket
            pass
        self.sock_to_prev.close()


class OutputNodeConnection(NodeConnection):
    """
    Class used to handle the output messages in each network node.
    The task of `OutputNodeConnection` objects is to extract messages from the output
    queue and send them to the next node in the chain over the output socket.

    The `run` method is supposed to run as a separate thread, allowing continuous
    operation of the processing thread (without blocking due to slow message
    transmission).

    Message are of JSON format (see `self.msg_format`).
    The execution loop will be automatically interrupted when receiving a message with
    the `"stop"` field set to True.
    """

    def __init__(
        self,
        config: Dict,
        next_node: Dict[str, Any],
        queue: deque,
        event_callback: threading.Event,
        max_tries: int = 30,
        **kwargs,
    ):
        """
        Initialize output socket (used to transmit the activations to the next node).
        Note: the output socket acts as a client - need the info of the next node!

        Args:
            config: dict containing (necessarily) the keys:
                addr: IP address of the node
                inference:
                    port_out: port for the socket
            queue: input queue of the node - used to pass the messages between this
                class and GPTServer
            event_callback: event that needs to be 'set' when queue is NOT empty
            max_tries (default: 30): maximum number of tries for setting up the
                connection
            **kwargs
        """
        super().__init__(**kwargs)

        self.name = "output_queue"
        self.message_queue = queue
        self.queue_not_empty = event_callback
        if len(self.message_queue):
            self.queue_not_empty.set()
        else:
            self.queue_not_empty.clear()
        self.config = config
        self.next_node_info = next_node
        self.sock_to_next = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connected = False
        tries = 0
        if self.verb:
            print("[INFO] Opening socket to next node")
        while not connected and tries < max_tries:
            try:
                self.sock_to_next = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # Bind should work even after some fails
                self.sock_to_next.bind(
                    (
                        self.config["addr"],
                        self.config["inference"]["port_out"],
                    )
                )
                self.sock_to_next.connect(
                    (
                        self.next_node_info["addr"],
                        self.next_node_info["inference"]["port_in"],
                    )
                )
                if self.verb:
                    print("Connected to next node!")
            except:
                # Can either fail when binding or when connecting
                tries += 1
                if self.verb:
                    print(
                        "> Unable to connect; retrying in 1s "
                        f"{loopsigns[tries % 4]}",
                        end="\r",
                    )
                time.sleep(1)
            else:
                connected = True

        if not connected:
            raise ConnectionError("Unable to connect to next node!")

        if self.verb:
            print("> Connection with next node established!               ")

    def send_msg(self, data: Any):
        """
        Send any Python object to the next node.
        The sender is a **client**.

        The `data` will be sent as raw bytes (pickled).

        The message is composed by a header of HEADERLENGTH bytes including the length
        of the actual message, plus a message of MSGLENGTH bytes containing the
        zero-padded message.
        """
        assert self.sock_to_next is not None

        message_str = pickle.dumps(data)
        tx_msg = bytes(f"{len(message_str):<{HEADERLENGTH}}", "utf-8") + message_str
        # NOTE: attempt at sending multiple messages in a "safe" way (no sendall)
        while tx_msg:
            tx_msg = tx_msg[self.sock_to_next.send(tx_msg) :]

    def run(self):
        while self.running.is_set():
            assert self.queue_not_empty.wait()

            tx_msg = self.message_queue.popleft()
            if len(self.message_queue) < 1:
                self.queue_not_empty.clear()

            if "stop" in tx_msg and tx_msg["stop"]:
                if self.verb:
                    print("[THR] Transmitting stopping message")
                self.send_msg(tx_msg)
                # FIXME - APP: should not stop everything
                self.running.clear()
            else:
                self.send_msg(tx_msg)

        if self.verb:
            print("[THR] Output queue thread stopped")

    def shutdown(self):
        self.running.clear()
        self.running_thread.join()
        if self.verb:
            print("[THR] Closing output socket")
        self.sock_to_next.close()
