using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>Continuous first-person motion with relative BCI turns on a DFS grid.</summary>
[DisallowMultipleComponent]
public sealed class MazeGameController : MonoBehaviour
{
    public static MazeGameController Instance { get; private set; }
    public enum TravelState { Moving, Turning, Junction, DeadEnd, Complete }
    [Header("Maze")]
    [Range(7, 61)] public int mazeWidth = 15;
    [Range(7, 61)] public int mazeHeight = 15;
    public bool useFixedSeed;
    public int fixedSeed = 20260920;
    [Header("First person")]
    [Min(1f)] public float cellSize = 2.8f;
    [Min(0.1f)] public float moveSpeed = 2.6f;
    [Min(0.05f)] public float turnSeconds = 0.45f;
    [Range(60f, 100f)] public float fieldOfView = 78f;
    public float eyeHeight = 1.6f;
    [Header("Junctions")]
    [Min(0f)] public float junctionWaitSeconds = 4f;
    public bool autoUTurnAtDeadEnd = true;
    [Min(0f)] public float deadEndWaitSeconds = 2f;
    public bool showMinimap = true;

    public TravelState State { get; private set; }
    public Vector2Int PlayerCell { get; private set; }
    public Vector2Int GoalCell { get; private set; }
    public int Heading { get; private set; }
    public int Steps { get; private set; }
    public int Turns { get; private set; }
    public int MazeGeneration { get; private set; }
    public bool Paused { get; private set; }
    public float ElapsedSeconds { get; private set; }
    public float DecisionRemaining => State == TravelState.Junction || State == TravelState.DeadEnd
        ? Mathf.Max(0, (State == TravelState.DeadEnd ? deadEndWaitSeconds : junctionWaitSeconds) - decisionElapsed) : 0;
    public string PendingTurn { get; private set; } = "";
    public string Hint { get; private set; } = "";
    public Texture2D Minimap { get; private set; }
    public bool CanLeft => IsWalkable(PlayerCell + Directions[(Heading + 3) % 4]);
    public bool CanRight => IsWalkable(PlayerCell + Directions[(Heading + 1) % 4]);
    public bool CanForward => IsWalkable(PlayerCell + Directions[Heading]);
    public bool CanUTurn => IsWalkable(PlayerCell + Directions[(Heading + 2) % 4]);
    public string HeadingName => new[] { "N", "E", "S", "W" }[Heading];
    // Grid +Y is world +Z; heading indices rotate clockwise.
    public static readonly Vector2Int[] Directions = {
        Vector2Int.up, Vector2Int.right, Vector2Int.down, Vector2Int.left
    };
    private bool[,] walkable;
    private Camera view;
    private MazeWorldRenderer world;
    private Vector2Int destination;
    private Quaternion turnFrom, turnTo;
    private float turnElapsed, decisionElapsed;
    private IMECompositionMode previousImeMode;

    private void Awake()
    {
        Instance = this;
        Application.runInBackground = true;
        previousImeMode = Input.imeCompositionMode;
        Input.imeCompositionMode = IMECompositionMode.Off;
    }
    private void Start()
    {
        world = gameObject.AddComponent<MazeWorldRenderer>();
        gameObject.AddComponent<MazeHud>();
        GenerateMaze();
    }

    public void GenerateMaze()
    {
        MazeGeneration++;
        mazeWidth = Mathf.Clamp(mazeWidth | 1, 7, 61);
        mazeHeight = Mathf.Clamp(mazeHeight | 1, 7, 61);
        walkable = new bool[mazeWidth, mazeHeight];
        var random = useFixedSeed ? new System.Random(fixedSeed) : new System.Random();
        var stack = new Stack<Vector2Int>();
        var start = new Vector2Int(1, 1);
        stack.Push(start);
        walkable[1, 1] = true;
        while (stack.Count > 0)
        {
            var current = stack.Peek();
            var candidates = new List<Vector2Int>();
            foreach (var direction in Directions)
            {
                var next = current + direction * 2;
                if (next.x > 0 && next.y > 0 && next.x < mazeWidth - 1 && next.y < mazeHeight - 1
                    && !walkable[next.x, next.y]) candidates.Add(direction);
            }
            if (candidates.Count == 0) { stack.Pop(); continue; }
            var chosen = candidates[random.Next(candidates.Count)];
            var nextCell = current + chosen * 2;
            walkable[current.x + chosen.x, current.y + chosen.y] = true;
            walkable[nextCell.x, nextCell.y] = true;
            stack.Push(nextCell);
        }
        PlayerCell = start;
        GoalCell = new Vector2Int(mazeWidth - 2, mazeHeight - 2);
        Heading = IsWalkable(start + Directions[0]) ? 0 : 1;
        Steps = Turns = 0;
        ElapsedSeconds = 0;
        decisionElapsed = 0;
        Paused = false;
        PendingTurn = Hint = "";
        if (world == null) world = GetComponent<MazeWorldRenderer>();
        world.Build(this);
        BuildMinimap();
        view = Camera.main;
        view.orthographic = false;
        view.fieldOfView = fieldOfView;
        view.nearClipPlane = 0.05f;
        view.farClipPlane = 250f;
        view.clearFlags = CameraClearFlags.SolidColor;
        view.backgroundColor = new Color(0.66f, 0.82f, 0.89f);
        view.transform.SetPositionAndRotation(EyePosition(start), Quaternion.Euler(0, Heading * 90, 0));
        BeginMove();
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.A) || Input.GetKeyDown(KeyCode.LeftArrow)) TryMove("left");
        if (Input.GetKeyDown(KeyCode.D) || Input.GetKeyDown(KeyCode.RightArrow)) TryMove("right");
        if (Input.GetKeyDown(KeyCode.S) || Input.GetKeyDown(KeyCode.DownArrow)) TryMove("uturn");
        if (Input.GetKeyDown(KeyCode.W) || Input.GetKeyDown(KeyCode.UpArrow)) TryMove("forward");
        if (Input.GetKeyDown(KeyCode.Space)) TogglePause();
        if (Input.GetKeyDown(KeyCode.R)) GenerateMaze();
        if (Input.GetKeyDown(KeyCode.M) || Input.GetKeyDown(KeyCode.V)) showMinimap = !showMinimap;
        if (Input.GetKeyDown(KeyCode.H)) ShowHint();
        TickNavigation(Time.deltaTime);
    }

    public void TickNavigation(float deltaTime)
    {
        if (walkable == null || Paused || State == TravelState.Complete) return;
        ElapsedSeconds += deltaTime;
        if (State == TravelState.Moving)
        {
            view.transform.position = Vector3.MoveTowards(view.transform.position, EyePosition(destination), moveSpeed * deltaTime);
            if (Vector3.Distance(view.transform.position, EyePosition(destination)) < 0.001f)
            {
                PlayerCell = destination;
                Steps++;
                Arrive();
            }
        }
        else if (State == TravelState.Turning)
        {
            turnElapsed += deltaTime;
            float t = Mathf.Clamp01(turnElapsed / Mathf.Max(0.05f, turnSeconds));
            view.transform.rotation = Quaternion.Slerp(turnFrom, turnTo, t * t * (3 - 2 * t));
            if (t >= 1) BeginMove();
        }
        else
        {
            if (ApplyPendingTurn()) return;
            decisionElapsed += deltaTime;
            if (State == TravelState.Junction && CanForward && decisionElapsed >= junctionWaitSeconds) BeginMove();
            else if (State == TravelState.DeadEnd && autoUTurnAtDeadEnd && decisionElapsed >= deadEndWaitSeconds) BeginTurn(2);
        }
    }

    private void Arrive()
    {
        if (PlayerCell == GoalCell) { State = TravelState.Complete; PendingTurn = ""; return; }
        if (ApplyPendingTurn()) return;
        int exits = (CanForward ? 1 : 0) + (CanLeft ? 1 : 0) + (CanRight ? 1 : 0);
        if (exits == 1)
        {
            if (CanForward) BeginMove();
            else BeginTurn(CanLeft ? -1 : 1);
        }
        else { State = exits == 0 ? TravelState.DeadEnd : TravelState.Junction; decisionElapsed = 0; }
    }

    private void BeginMove() { destination = PlayerCell + Directions[Heading]; State = TravelState.Moving; }
    private void BeginTurn(int quarterTurns)
    {
        Heading = (Heading + quarterTurns + 4) % 4;
        Turns++;
        turnFrom = view.transform.rotation;
        turnTo = Quaternion.Euler(0, Heading * 90, 0);
        turnElapsed = 0;
        State = TravelState.Turning;
    }
    private bool ApplyPendingTurn()
    {
        int turn = PendingTurn == "left" ? -1 : PendingTurn == "right" ? 1 : 2;
        bool available = PendingTurn == "left" ? CanLeft : PendingTurn == "right" ? CanRight : CanUTurn;
        if (PendingTurn.Length == 0 || !available) return false;
        PendingTurn = "";
        BeginTurn(turn);
        return true;
    }

    /// <summary>Main-thread endpoint. Accepted turns can wait for the next available exit.</summary>
    public bool TryMove(string command)
    {
        string value = NormalizeCommand(command);
        if (value == "restart") { GenerateMaze(); return true; }
        if (value == "pause") { Paused = true; return true; }
        if (value == "resume") { Paused = false; return true; }
        if (value == "cancel") { PendingTurn = ""; return true; }
        if (walkable == null || State == TravelState.Complete) return false;
        if (value == "forward")
        {
            PendingTurn = "";
            Paused = false;
            if (State == TravelState.Junction && CanForward) BeginMove();
            return true;
        }
        if (value != "left" && value != "right" && value != "uturn") return false;
        PendingTurn = value;
        if (!Paused && (State == TravelState.Junction || State == TravelState.DeadEnd)) ApplyPendingTurn();
        return true;
    }
    public static bool SubmitExternalCommand(string command) => Instance != null && Instance.TryMove(command);
    public bool MatchesExternalState(long generation, long x, long y, long heading, long steps, bool paused, string state)
        => MazeGeneration == generation && PlayerCell.x == x && PlayerCell.y == y
        && Heading == heading && Steps == steps && Paused == paused && State.ToString() == state;
    public void TogglePause() => Paused = !Paused;
    public static string NormalizeCommand(string raw)
    {
        switch ((raw ?? "").Trim().ToLowerInvariant())
        {
            case "left": case "turn_left": case "move_left": case "a": case "west": return "left";
            case "right": case "turn_right": case "move_right": case "d": case "east": return "right";
            case "uturn": case "u_turn": case "turn_around": case "back": case "backward":
            case "down": case "move_down": case "south": case "s": return "uturn";
            case "forward": case "up": case "move_up": case "north": case "w": return "forward";
            case "stop": case "pause": return "pause";
            case "resume": return "resume";
            case "cancel": return "cancel";
            case "restart": return "restart";
            case "heartbeat": return "heartbeat";
            case "status": return "status";
            default: return "";
        }
    }
    public bool IsWalkable(Vector2Int cell) => walkable != null && cell.x >= 0 && cell.y >= 0
        && cell.x < mazeWidth && cell.y < mazeHeight && walkable[cell.x, cell.y];
    public Vector3 CellToWorld(Vector2Int cell) => new Vector3(cell.x * cellSize, 0, cell.y * cellSize);
    private Vector3 EyePosition(Vector2Int cell) => CellToWorld(cell) + Vector3.up * eyeHeight;

    public IReadOnlyList<Vector2Int> GetHintPath()
    {
        var queue = new Queue<Vector2Int>();
        var parents = new Dictionary<Vector2Int, Vector2Int>();
        queue.Enqueue(PlayerCell);
        parents[PlayerCell] = PlayerCell;
        while (queue.Count > 0)
        {
            var current = queue.Dequeue();
            if (current == GoalCell) break;
            foreach (var direction in Directions)
            {
                var next = current + direction;
                if (!IsWalkable(next) || parents.ContainsKey(next)) continue;
                parents[next] = current;
                queue.Enqueue(next);
            }
        }
        if (!parents.ContainsKey(GoalCell)) return Array.Empty<Vector2Int>();
        var path = new List<Vector2Int>();
        for (var cell = GoalCell; cell != PlayerCell; cell = parents[cell]) path.Add(cell);
        path.Add(PlayerCell);
        path.Reverse();
        return path;
    }
    public void ShowHint()
    {
        var path = GetHintPath();
        if (path.Count < 2) { Hint = "Exit reached"; return; }
        int direction = Array.IndexOf(Directions, path[1] - path[0]);
        Hint = new[] { "Continue ahead", "Next exit: right", "Turn around", "Next exit: left" }[(direction - Heading + 4) % 4];
    }
    public object GetStatus() => new {
        mode = "first_person_relative", maze_generation = MazeGeneration, state = State.ToString(), paused = Paused,
        x = PlayerCell.x, y = PlayerCell.y, heading = Heading, heading_name = HeadingName,
        steps = Steps, turns = Turns, pending_turn = PendingTurn, remaining_seconds = DecisionRemaining,
        can_left = CanLeft, can_right = CanRight, can_forward = CanForward, can_uturn = CanUTurn,
        goal_x = GoalCell.x, goal_y = GoalCell.y, elapsed_seconds = ElapsedSeconds
    };
    private void BuildMinimap()
    {
        if (Minimap != null) Destroy(Minimap);
        Minimap = new Texture2D(mazeWidth, mazeHeight) { filterMode = FilterMode.Point };
        for (int x = 0; x < mazeWidth; x++)
        for (int y = 0; y < mazeHeight; y++)
            Minimap.SetPixel(x, y, walkable[x, y] ? new Color(0.76f, 0.83f, 0.83f) : new Color(0.08f, 0.16f, 0.18f));
        Minimap.SetPixel(1, 1, new Color(0.14f, 0.78f, 0.65f));
        Minimap.SetPixel(GoalCell.x, GoalCell.y, new Color(1f, 0.55f, 0.25f));
        Minimap.Apply();
    }
    private void OnDestroy()
    {
        if (Instance == this) Instance = null;
        Input.imeCompositionMode = previousImeMode;
        if (Minimap != null) Destroy(Minimap);
    }
}
