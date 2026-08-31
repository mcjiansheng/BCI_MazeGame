using System;
using System.Collections.Concurrent;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// Thread-safe external command adapter. It accepts UTF-8 UDP commands on the
/// configured loopback port and queues them for MazeGameController on Unity's main thread.
///
/// Payloads are either exact command tokens ("left") or flat JSON objects
/// ({"command":"left","ts":123.4}). Substring matching is deliberately avoided
/// so payloads such as {"status":"backup"} can never trigger a movement.
/// A "heartbeat" command keeps the link status alive without moving the player.
/// </summary>
[DisallowMultipleComponent]
public sealed class ExternalMazeInput : MonoBehaviour
{
    public static ExternalMazeInput Instance { get; private set; }

    [Header("UDP BCI bridge")]
    [Min(1024)] public int udpPort = 7777;
    public bool listenOnLoopbackOnly = true;
    [Range(1, 32)] public int maxCommandsPerFrame = 4;
    [Range(1f, 60f)] public float linkStaleSeconds = 5f;
    public int ReceivedUdpCommandCount => Volatile.Read(ref receivedUdpCommandCount);
    public int ProcessedCommandCount => Volatile.Read(ref processedCommandCount);

    /// <summary>Human-readable inference link state for HUD display.</summary>
    public string LinkStatus
    {
        get
        {
            long ticks = Interlocked.Read(ref lastPacketUtcTicks);
            if (ticks == 0) return "BCI bridge: waiting for packets";
            double ageSeconds = (DateTime.UtcNow.Ticks - ticks) / (double)TimeSpan.TicksPerSecond;
            return ageSeconds <= linkStaleSeconds
                ? $"BCI bridge: online ({ageSeconds:F1}s ago)"
                : $"BCI bridge: stale ({ageSeconds:F0}s ago)";
        }
    }

    private readonly ConcurrentQueue<ReceivedCommand> pendingCommands = new ConcurrentQueue<ReceivedCommand>();
    private UdpClient client;
    private Thread receiveThread;
    private volatile bool receiving;
    private long lastPacketUtcTicks;
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
            if (received.replyTo != null) SendAcknowledgement(received.replyTo, received.command, accepted, null);
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
        if (string.IsNullOrEmpty(normalized) || normalized == "heartbeat") return false;
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
                string payload = Encoding.UTF8.GetString(bytes);
                if (string.IsNullOrWhiteSpace(payload)) continue;
                // Any datagram from the bridge proves the link is alive, even
                // if the payload itself is not a movement command.
                Interlocked.Exchange(ref lastPacketUtcTicks, DateTime.UtcNow.Ticks);

                string command = NormalizePayload(payload);
                string echoTimestamp = ExtractJsonRawValue(payload, "ts");
                if (command == "heartbeat")
                {
                    SendAcknowledgement(new IPEndPoint(remote.Address, remote.Port), command, true, echoTimestamp);
                    continue;
                }
                if (string.IsNullOrEmpty(command)) continue;

                // UDP runs on a worker thread. The maze state update is protected by
                // MazeGameController; the visible player is synchronized in LateUpdate.
                bool accepted = MazeGameController.SubmitExternalCommand(command);
                SendAcknowledgement(new IPEndPoint(remote.Address, remote.Port), command, accepted, echoTimestamp);
                Interlocked.Increment(ref receivedUdpCommandCount);
                Interlocked.Increment(ref processedCommandCount);
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
        string value = (payload ?? string.Empty).Trim();
        if (value.Length == 0) return string.Empty;
        if (value[0] == '{')
            return NormalizeToken(ExtractJsonStringField(value, "command"));
        return NormalizeToken(value);
    }

    private static string NormalizeToken(string token)
    {
        switch ((token ?? string.Empty).Trim().ToLowerInvariant())
        {
            case "up": case "north": case "forward": case "w": case "move_up": return "up";
            case "down": case "south": case "back": case "backward": case "s": case "move_down": return "down";
            case "left": case "west": case "a": case "move_left": return "left";
            case "right": case "east": case "d": case "move_right": return "right";
            case "heartbeat": return "heartbeat";
            default: return string.Empty;
        }
    }

    /// <summary>Extracts a string field from a flat JSON payload without allocation-heavy parsing.</summary>
    private static string ExtractJsonStringField(string json, string field)
    {
        int fieldIndex = json.IndexOf($"\"{field}\"", StringComparison.OrdinalIgnoreCase);
        if (fieldIndex < 0) return null;
        int colonIndex = json.IndexOf(':', fieldIndex + field.Length + 2);
        if (colonIndex < 0) return null;
        int start = json.IndexOf('"', colonIndex + 1);
        if (start < 0) return null;
        int end = json.IndexOf('"', start + 1);
        if (end < 0) return null;
        return json.Substring(start + 1, end - start - 1);
    }

    /// <summary>Extracts a raw (unquoted) numeric field, e.g. "ts", for echoing back.</summary>
    private static string ExtractJsonRawValue(string json, string field)
    {
        int fieldIndex = json.IndexOf($"\"{field}\"", StringComparison.OrdinalIgnoreCase);
        if (fieldIndex < 0) return null;
        int colonIndex = json.IndexOf(':', fieldIndex + field.Length + 2);
        if (colonIndex < 0) return null;
        var builder = new StringBuilder();
        for (int index = colonIndex + 1; index < json.Length; index++)
        {
            char character = json[index];
            if (char.IsWhiteSpace(character) && builder.Length == 0) continue;
            if (character == ',' || character == '}') break;
            // Only numeric characters are echoed; anything else aborts the value.
            if (
                !char.IsDigit(character)
                && character != '.'
                && character != '-'
                && character != '+'
                && character != 'e'
                && character != 'E'
            ) break;
            builder.Append(character);
        }
        string rawValue = builder.ToString();
        if (
            !double.TryParse(
                rawValue,
                NumberStyles.Float,
                CultureInfo.InvariantCulture,
                out double parsed
            )
            || double.IsNaN(parsed)
            || double.IsInfinity(parsed)
        ) return null;
        return parsed.ToString("R", CultureInfo.InvariantCulture);
    }

    private static void SendAcknowledgement(IPEndPoint destination, string command, bool accepted, string echoTimestamp)
    {
        string timestampField = string.IsNullOrEmpty(echoTimestamp)
            ? string.Empty
            : $",\"echo_ts\":{echoTimestamp}";
        string json = $"{{\"command\":\"{command}\",\"accepted\":{accepted.ToString().ToLowerInvariant()}{timestampField}}}";
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
