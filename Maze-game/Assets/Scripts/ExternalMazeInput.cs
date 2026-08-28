using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// Thread-safe external command adapter. It accepts UTF-8 UDP commands on the
/// configured loopback port and queues them for MazeGameController on Unity's main thread.
/// </summary>
[DisallowMultipleComponent]
public sealed class ExternalMazeInput : MonoBehaviour
{
    public static ExternalMazeInput Instance { get; private set; }

    [Header("UDP BCI bridge")]
    [Min(1024)] public int udpPort = 7777;
    public bool listenOnLoopbackOnly = true;
    [Range(1, 32)] public int maxCommandsPerFrame = 4;
    public int ReceivedUdpCommandCount => Volatile.Read(ref receivedUdpCommandCount);
    public int ProcessedCommandCount => Volatile.Read(ref processedCommandCount);

    private readonly ConcurrentQueue<ReceivedCommand> pendingCommands = new ConcurrentQueue<ReceivedCommand>();
    private UdpClient client;
    private Thread receiveThread;
    private volatile bool receiving;
    private int receivedUdpCommandCount;
    private int processedCommandCount;

    private void Awake()
    {
        Instance = this;
    }

    private void OnEnable()
    {
        StartListener();
    }

    private void Update()
    {
        int processed = 0;
        while (processed < maxCommandsPerFrame && pendingCommands.TryDequeue(out ReceivedCommand received))
        {
            bool accepted = MazeGameController.SubmitExternalCommand(received.command);
            if (received.replyTo != null) SendAcknowledgement(received.replyTo, received.command, accepted);
            Interlocked.Increment(ref processedCommandCount);
            processed++;
        }
    }

    /// <summary>Use this from another Unity plugin (including a ZeroMQ bridge).</summary>
    public static bool SubmitCommand(string command)
    {
        if (string.IsNullOrWhiteSpace(command)) return false;
        if (Instance == null) return MazeGameController.SubmitExternalCommand(command);
        string normalized = NormalizePayload(command);
        if (string.IsNullOrEmpty(normalized)) return false;
        Instance.pendingCommands.Enqueue(new ReceivedCommand(normalized, null));
        return true;
    }

    private void StartListener()
    {
        try
        {
            IPAddress address = listenOnLoopbackOnly ? IPAddress.Loopback : IPAddress.Any;
            client = new UdpClient(new IPEndPoint(address, udpPort));
            receiving = true;
            receiveThread = new Thread(ReceiveLoop) { IsBackground = true, Name = "Maze UDP Input" };
            receiveThread.Start();
            Debug.Log($"Maze external input listening on udp://{address}:{udpPort}");
        }
        catch (Exception exception)
        {
            Debug.LogError($"Could not start the maze UDP input on port {udpPort}: {exception.Message}");
        }
    }

    private void ReceiveLoop()
    {
        var remote = new IPEndPoint(IPAddress.Any, 0);
        try
        {
            while (receiving)
            {
                byte[] bytes = client.Receive(ref remote);
                string command = NormalizePayload(Encoding.UTF8.GetString(bytes));
                if (!string.IsNullOrEmpty(command))
                {
                    // UDP runs on a worker thread. The maze state update is protected by
                    // MazeGameController; the visible player is synchronized in LateUpdate.
                    bool accepted = MazeGameController.SubmitExternalCommand(command);
                    SendAcknowledgement(new IPEndPoint(remote.Address, remote.Port), command, accepted);
                    Interlocked.Increment(ref receivedUdpCommandCount);
                    Interlocked.Increment(ref processedCommandCount);
                }
            }
        }
        catch (SocketException) when (!receiving) { }
        catch (ObjectDisposedException) when (!receiving) { }
        catch (Exception exception)
        {
            if (receiving) Debug.LogError($"Maze UDP receiver stopped: {exception.Message}");
        }
    }

    private static string NormalizePayload(string payload)
    {
        string value = (payload ?? string.Empty).Trim().ToLowerInvariant();
        // Accept both plain text ("left") and lightweight JSON ("{\"command\":\"left\"}").
        if (value.Contains("up") || value.Contains("forward") || value == "w") return "up";
        if (value.Contains("down") || value.Contains("back") || value == "s") return "down";
        if (value.Contains("left") || value == "a") return "left";
        if (value.Contains("right") || value == "d") return "right";
        return string.Empty;
    }

    private static void SendAcknowledgement(IPEndPoint destination, string command, bool accepted)
    {
        string json = $"{{\"command\":\"{command}\",\"accepted\":{accepted.ToString().ToLowerInvariant()}}}";
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        using (var response = new UdpClient()) response.Send(bytes, bytes.Length, destination);
    }

    private readonly struct ReceivedCommand
    {
        public readonly string command;
        public readonly IPEndPoint replyTo;

        public ReceivedCommand(string command, IPEndPoint replyTo)
        {
            this.command = command;
            this.replyTo = replyTo;
        }
    }

    private void OnDisable()
    {
        receiving = false;
        client?.Close();
        if (receiveThread != null && receiveThread.IsAlive) receiveThread.Join(250);
        receiveThread = null;
        client = null;
    }
}
