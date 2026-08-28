# BCI Maze game

Press **Play** in Unity to generate a new 61×61 random perfect maze. The map uses iterative DFS (recursive backtracking), so every passable cell is connected and the exit has a unique solution path. Keyboard controls are **WASD** or the arrow keys; **R** makes a new maze and **H** logs the remaining BFS shortest-path length.

The default camera is a synchronized top-down player view: it remains centred on the player after every keyboard or external movement command. Press **V** to switch between that player view and the full-map overview. The `BCI Maze Game` object's `MazeGameController` component exposes the follow flag and player-view zoom in the Inspector.

## External movement interface

The `ExternalMazeInput` component listens only on `127.0.0.1:7777` by default, keeping the BCI bridge local to the machine. Send one UTF-8 UDP datagram for each intended step:

```text
up | down | left | right
```

It also accepts a lightweight JSON payload such as `{"command":"left"}`. The receiver queues data on its background thread, then applies movement safely on Unity's main thread with the 120 ms anti-bounce cooldown described in the project documentation.

For request/response integration tests, Unity replies to the UDP sender with `{"command":"left","accepted":true}` only when that model-derived move is legal and has been applied to the player.

Python test sender:

```python
import socket
socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"right", ("127.0.0.1", 7777))
```

An in-process BCI/ZeroMQ bridge can instead call `ExternalMazeInput.SubmitCommand("left")` or `MazeGameController.SubmitExternalCommand("left")`.

## Real EEG end-to-end test

With Unity playing and the UDP receiver active, run:

```powershell
.\.venv\Scripts\python.exe scripts\run_bci_maze_integration.py
```

The test loads an actual artifact-filtered BCIC IV 2a A01 evaluation trial, normalizes it with the same subject-training statistics used by the stored `SE-MHAF-Final` checkpoint, obtains the predicted MI class and confidence, maps it to a maze command, waits for Unity's UDP acknowledgement, and writes a reproducible report to `outputs/bci_maze_integration_test.json`.
