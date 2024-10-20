import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests


class MessageModule:
    """
    Module used to query the HTTP API.
    """

    docstring = """
    Available commands:
        - help: print this message
        - list models: list available models
        - list nodes: list all available nodes
        - run <model_name>: launch specific models
        - stop
    """
    valid_commands = {"help", "list models", "list nodes", "run", "stop"}

    def __init__(self, address: str, port: int):
        """ """
        self.serv_addr = address
        self.serv_port = port

    def start(self, stop_callback: Callable):
        """Begin operation"""
        while True:
            cmd = str(input(">>> "))

            if cmd == "help":
                print(self.docstring)
            elif cmd == "list models":
                # GET to server
                resp = json.loads(
                    requests.get(f"{self.serv_addr}:{self.serv_port}").json()
                )
                models = resp["model"]["available"]
                if not len(models):
                    print("No available model!")
                else:
                    print("Available models:")
                    for mod_dict in models:
                        hf_config = mod_dict["hf_config"]
                        print(f"\t{hf_config['org']}/{hf_config['name']}")
            elif cmd == "list models":
                pass
            elif cmd == "list nodes":
                pass
            elif cmd == "run":
                pass
            elif cmd == "stop":
                pass
                break
            else:
                print("Invalid command!")
                print(self.docstring)
