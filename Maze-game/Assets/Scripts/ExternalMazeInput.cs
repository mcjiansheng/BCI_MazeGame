using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

/// <summary>UDP receives on a worker; navigation and ACKs run on Unity's main thread.</summary>
[DisallowMultipleComponent]
public sealed class ExternalMazeInput : MonoBehaviour
{
    public static ExternalMazeInput Instance { get; private set; }
    public int udpPort = 7777;
    public bool listenOnLoopbackOnly = true;
    public int maxCommandsPerFrame = 4;
    public float linkStaleSeconds = 5f;
    public int ReceivedUdpCommandCount => received;
    public int ProcessedCommandCount => processed;
    public string LinkStatus => lastPacketTicks == 0 ? "BCI waiting"
        : (DateTime.UtcNow.Ticks - Interlocked.Read(ref lastPacketTicks)) / (double)TimeSpan.TicksPerSecond < linkStaleSeconds
            ? "BCI connected" : "BCI idle";
    private struct Command { public string payload; public IPEndPoint remote; }
    private readonly ConcurrentQueue<Command> pending = new ConcurrentQueue<Command>();
    private UdpClient client;
    private Thread receiver;
    private volatile bool running;
    private int received, processed;
    private long lastPacketTicks;

    private void OnEnable()
    {
        Instance = this;
        try
        {
            client = new UdpClient(new IPEndPoint(listenOnLoopbackOnly ? IPAddress.Loopback : IPAddress.Any, udpPort));
            running = true;
            receiver = new Thread(Receive) { IsBackground = true, Name = "Maze UDP" };
            receiver.Start();
            Debug.Log($"First-person maze UDP listening on {udpPort}");
        }
        catch (SocketException error) { Debug.LogWarning($"Maze UDP: {error.Message}"); }
    }
    private void Receive()
    {
        var remote = new IPEndPoint(IPAddress.Any, 0);
        try
        {
            while (running)
            {
                var bytes = client.Receive(ref remote);
                Interlocked.Exchange(ref lastPacketTicks, DateTime.UtcNow.Ticks);
                Interlocked.Increment(ref received);
                pending.Enqueue(new Command { payload = Encoding.UTF8.GetString(bytes), remote = new IPEndPoint(remote.Address, remote.Port) });
            }
        }
        catch (SocketException) when (!running) { }
        catch (ObjectDisposedException) when (!running) { }
        catch (Exception error) { Debug.LogWarning($"Maze UDP stopped: {error.Message}"); }
    }
    private void Update()
    {
        for (int i = 0; i < maxCommandsPerFrame && pending.TryDequeue(out var item); i++)
        {
            string raw = item.payload.Trim();
            JToken timestamp = null;
            JToken expected = null;
            try
            {
                if (raw.StartsWith("{"))
                {
                    var json = JObject.Parse(raw);
                    raw = (string)json["command"] ?? "";
                    timestamp = json["ts"];
                    expected = json["expected"];
                }
            }
            catch (JsonException) { raw = ""; }
            string command = MazeGameController.NormalizeCommand(raw);
            // Validation and navigation run consecutively on Unity's main thread.
            // A restart, resume, or move between status and command invalidates the old decision.
            bool expectedMatches = expected == null || MatchesExpected(expected);
            bool accepted = expectedMatches && (command == "heartbeat" || command == "status"
                || (command.Length > 0 && MazeGameController.SubmitExternalCommand(command)));
            processed++;
            if (item.remote == null) continue;
            var response = new JObject { ["command"] = command, ["accepted"] = accepted };
            if (timestamp != null) response["echo_ts"] = timestamp;
            if (MazeGameController.Instance != null)
                response["navigation"] = JObject.FromObject(MazeGameController.Instance.GetStatus());
            byte[] bytes = Encoding.UTF8.GetBytes(response.ToString(Formatting.None));
            try { client.Send(bytes, bytes.Length, item.remote); }
            catch (SocketException error) { Debug.LogWarning($"Maze ACK: {error.Message}"); }
        }
    }
    private static bool MatchesExpected(JToken token)
    {
        var expected = token as JObject;
        var game = MazeGameController.Instance;
        if (expected == null || game == null) return false;
        foreach (string key in new[] { "maze_generation", "x", "y", "heading", "steps" })
            if (expected[key]?.Type != JTokenType.Integer) return false;
        if (expected["paused"]?.Type != JTokenType.Boolean || expected["state"]?.Type != JTokenType.String) return false;
        if (expected["pending_turn"] != null && (expected["pending_turn"].Type != JTokenType.String
            || (string)expected["pending_turn"] != game.PendingTurn)) return false;
        try
        {
            return game.MatchesExternalState((long)expected["maze_generation"], (long)expected["x"],
                (long)expected["y"], (long)expected["heading"], (long)expected["steps"],
                (bool)expected["paused"], (string)expected["state"]);
        }
        catch (Exception error) when (error is OverflowException || error is FormatException || error is InvalidCastException)
        {
            return false;
        }
    }
    /// <summary>Thread-safe plugin entry point; true means enqueued, not yet executed.</summary>
    public static bool SubmitCommand(string command)
    {
        if (Instance == null || string.IsNullOrWhiteSpace(command)) return false;
        Instance.pending.Enqueue(new Command { payload = command });
        return true;
    }
    private void OnDisable()
    {
        running = false;
        client?.Close();
        receiver?.Join(250);
        client = null;
        while (pending.TryDequeue(out _)) { }
        if (Instance == this) Instance = null;
    }
}
