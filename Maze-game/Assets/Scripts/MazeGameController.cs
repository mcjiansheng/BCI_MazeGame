using System;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Generates and runs a top-down maze.  The maze uses iterative DFS with a
/// one-cell wall between passages, as described in the project documentation.
/// </summary>
[DisallowMultipleComponent]
public sealed class MazeGameController : MonoBehaviour
{
    public static MazeGameController Instance { get; private set; }
    private static SynchronizationContext mainThreadContext;

    [Header("Maze")]
    [Range(21, 101)] public int mazeWidth = 61;
    [Range(21, 101)] public int mazeHeight = 61;
    [Min(0.05f)] public float moveCooldownSeconds = 0.12f;
    public bool useFixedSeed;
    public int fixedSeed = 20260723;

    [Header("Colours")]
    public Color wallColor = new Color(0.07f, 0.12f, 0.22f);
    public Color floorColor = new Color(0.82f, 0.88f, 0.93f);
    public Color startColor = new Color(0.15f, 0.72f, 0.38f);
    public Color goalColor = new Color(0.90f, 0.24f, 0.28f);
    public Color playerColor = new Color(1.00f, 0.78f, 0.10f);

    [Header("Camera")]
    [Tooltip("Keep the player centred while moving through the maze.")]
    public bool followPlayerCamera = true;
    [Min(3f)] public float playerViewSize = 9f;

    private bool[,] walkable;
    private Vector2Int startCell;
    private Vector2Int goalCell;
    private Vector2Int playerCell;
    private SpriteRenderer mapRenderer;
    private SpriteRenderer playerRenderer;
    private Texture2D mazeTexture;
    private float lastMoveTime = -999f;
    private bool hasWon;
    private int steps;
    private string status = "Generating maze…";
    private Camera gameCamera;
    private readonly object mazeStateLock = new object();
    private bool pendingExternalVisualUpdate;

    private static readonly Vector2Int[] CarveDirections =
    {
        new Vector2Int(2, 0),
        new Vector2Int(-2, 0),
        new Vector2Int(0, 2),
        new Vector2Int(0, -2)
    };

    private static readonly Vector2Int[] MoveDirections =
    {
        Vector2Int.up,
        Vector2Int.down,
        Vector2Int.left,
        Vector2Int.right
    };

    private void Awake()
    {
        Instance = this;
        mainThreadContext = SynchronizationContext.Current;
    }

    private void OnEnable()
    {
        // Also register after an editor/domain reload, where Awake may not run again
        // for an already-active scene object.
        Instance = this;
        mainThreadContext = SynchronizationContext.Current;
#if UNITY_EDITOR
        EditorApplication.update -= EditorVisualSync;
        EditorApplication.update += EditorVisualSync;
        EditorApplication.delayCall -= EnsureMazeInitialized;
        EditorApplication.delayCall += EnsureMazeInitialized;
#endif
    }

    private void OnDisable()
    {
#if UNITY_EDITOR
        EditorApplication.update -= EditorVisualSync;
        EditorApplication.delayCall -= EnsureMazeInitialized;
#endif
    }

#if UNITY_EDITOR
    // The Unity Editor may remain in a play-mode transition while its MCP bridge
    // reconnects. This editor-main-thread callback keeps externally requested
    // movement visually synchronized during that short transition as well.
    private void EditorVisualSync()
    {
        SynchronizeExternalVisuals();
        if (followPlayerCamera) UpdateCameraPosition();
    }

    private void EnsureMazeInitialized()
    {
        if (Application.isPlaying && walkable == null) GenerateMaze();
    }
#endif

    private void Start()
    {
        GenerateMaze();
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.W) || Input.GetKeyDown(KeyCode.UpArrow)) TryMove("up");
        if (Input.GetKeyDown(KeyCode.S) || Input.GetKeyDown(KeyCode.DownArrow)) TryMove("down");
        if (Input.GetKeyDown(KeyCode.A) || Input.GetKeyDown(KeyCode.LeftArrow)) TryMove("left");
        if (Input.GetKeyDown(KeyCode.D) || Input.GetKeyDown(KeyCode.RightArrow)) TryMove("right");
        if (Input.GetKeyDown(KeyCode.R)) GenerateMaze();
        if (Input.GetKeyDown(KeyCode.H)) ShowHintInConsole();
        if (Input.GetKeyDown(KeyCode.V)) ToggleCameraView();
    }

    private void LateUpdate()
    {
        SynchronizeExternalVisuals();
        if (followPlayerCamera) UpdateCameraPosition();
    }

    private void SynchronizeExternalVisuals()
    {
        bool updateVisual;
        Vector2Int externalPlayerCell;
        lock (mazeStateLock)
        {
            updateVisual = pendingExternalVisualUpdate;
            externalPlayerCell = playerCell;
            pendingExternalVisualUpdate = false;
        }
        if (updateVisual && playerRenderer != null)
            playerRenderer.transform.position = GridToWorld(externalPlayerCell);
        if (updateVisual && followPlayerCamera) UpdateCameraPosition();
    }

    /// <summary>Creates a new connected perfect maze and resets the player.</summary>
    public void GenerateMaze()
    {
        mazeWidth = MakeOdd(mazeWidth);
        mazeHeight = MakeOdd(mazeHeight);
        walkable = new bool[mazeWidth, mazeHeight];

        var random = useFixedSeed ? new System.Random(fixedSeed) : new System.Random();
        var stack = new Stack<Vector2Int>();
        startCell = new Vector2Int(1, 1);
        stack.Push(startCell);
        walkable[startCell.x, startCell.y] = true;

        while (stack.Count > 0)
        {
            Vector2Int current = stack.Peek();
            var candidates = new List<Vector2Int>(4);
            foreach (Vector2Int direction in CarveDirections)
            {
                Vector2Int next = current + direction;
                if (IsInterior(next) && !walkable[next.x, next.y]) candidates.Add(direction);
            }

            if (candidates.Count == 0)
            {
                stack.Pop();
                continue;
            }

            Vector2Int chosen = candidates[random.Next(candidates.Count)];
            Vector2Int passage = current + chosen / 2;
            Vector2Int nextCell = current + chosen;
            walkable[passage.x, passage.y] = true;
            walkable[nextCell.x, nextCell.y] = true;
            stack.Push(nextCell);
        }

        goalCell = new Vector2Int(mazeWidth - 2, mazeHeight - 2);
        playerCell = startCell;
        hasWon = false;
        steps = 0;
        lastMoveTime = -999f;
        BuildVisuals();
        ConfigureCamera();
        status = "Reach the red exit.";
        Debug.Log($"Generated {mazeWidth}×{mazeHeight} DFS maze. Start: {startCell}; goal: {goalCell}.");
    }

    /// <summary>
    /// Main-thread command endpoint for UI, BCI bridges and external adapters.
    /// Supported values: up/down/left/right, forward/backward, w/a/s/d.
    /// </summary>
    public bool TryMove(string command)
    {
        lock (mazeStateLock)
        {
            if (Time.unscaledTime - lastMoveTime < moveCooldownSeconds) return false;
            return MoveInternal(command, true);
        }
    }

    /// <summary>
    /// Thread-safe movement endpoint for network adapters. It updates the game state
    /// immediately, then LateUpdate synchronizes the rendered character on Unity's main thread.
    /// </summary>
    public bool TryMoveFromExternal(string command)
    {
        lock (mazeStateLock) return MoveInternal(command, false);
    }

    private bool MoveInternal(string command, bool updateVisualImmediately)
    {
        if (walkable == null || hasWon || !TryParseDirection(command, out Vector2Int direction)) return false;
        Vector2Int next = playerCell + direction;
        if (!IsWalkable(next))
        {
            status = "Wall blocked the move.";
            return false;
        }

        playerCell = next;
        steps++;
        if (updateVisualImmediately)
        {
            lastMoveTime = Time.unscaledTime;
            if (playerRenderer != null) playerRenderer.transform.position = GridToWorld(playerCell);
            UpdateCameraPosition();
        }
        else
        {
            pendingExternalVisualUpdate = true;
            QueueExternalVisualSync();
        }
        status = $"Steps: {steps}";

        if (playerCell == goalCell)
        {
            hasWon = true;
            status = $"Maze complete in {steps} steps. Press R for a new maze.";
            if (updateVisualImmediately) Debug.Log(status);
        }
        return true;
    }

    /// <summary>Call this from an in-process BCI/ZeroMQ bridge.</summary>
    public static bool SubmitExternalCommand(string command)
    {
        return Instance != null && Instance.TryMoveFromExternal(command);
    }

    private void QueueExternalVisualSync()
    {
        SynchronizationContext context = mainThreadContext;
        if (context != null) context.Post(_ => SynchronizeExternalVisuals(), null);
    }

    public IReadOnlyList<Vector2Int> GetHintPath()
    {
        if (walkable == null) return Array.Empty<Vector2Int>();
        var queue = new Queue<Vector2Int>();
        var visited = new bool[mazeWidth, mazeHeight];
        var parents = new Dictionary<Vector2Int, Vector2Int>();
        queue.Enqueue(playerCell);
        visited[playerCell.x, playerCell.y] = true;

        while (queue.Count > 0)
        {
            Vector2Int current = queue.Dequeue();
            if (current == goalCell) break;
            foreach (Vector2Int direction in MoveDirections)
            {
                Vector2Int next = current + direction;
                if (!IsWalkable(next) || visited[next.x, next.y]) continue;
                visited[next.x, next.y] = true;
                parents[next] = current;
                queue.Enqueue(next);
            }
        }

        if (!visited[goalCell.x, goalCell.y]) return Array.Empty<Vector2Int>();
        var path = new List<Vector2Int>();
        for (Vector2Int current = goalCell; current != playerCell; current = parents[current]) path.Add(current);
        path.Add(playerCell);
        path.Reverse();
        return path;
    }

    private void ShowHintInConsole()
    {
        IReadOnlyList<Vector2Int> path = GetHintPath();
        status = path.Count > 1 ? $"Hint: {path.Count - 1} moves remain (logged to Console)." : "Already at the exit.";
        Debug.Log(status);
    }

    private void BuildVisuals()
    {
        if (mapRenderer == null)
        {
            var mapObject = new GameObject("Maze Map");
            mapObject.transform.SetParent(transform, false);
            mapRenderer = mapObject.AddComponent<SpriteRenderer>();
            mapRenderer.sortingOrder = 0;
        }
        if (playerRenderer == null)
        {
            var playerObject = new GameObject("Maze Player");
            playerObject.transform.SetParent(transform, false);
            playerRenderer = playerObject.AddComponent<SpriteRenderer>();
            playerRenderer.sortingOrder = 2;
            playerRenderer.sprite = Sprite.Create(Texture2D.whiteTexture, new Rect(0f, 0f, 1f, 1f), new Vector2(0.5f, 0.5f), 1f);
            playerRenderer.color = playerColor;
            playerRenderer.transform.localScale = new Vector3(0.58f, 0.58f, 1f);
        }

        mazeTexture = new Texture2D(mazeWidth, mazeHeight, TextureFormat.RGBA32, false)
        {
            name = "RuntimeMazeTexture",
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        for (int x = 0; x < mazeWidth; x++)
        for (int y = 0; y < mazeHeight; y++)
            mazeTexture.SetPixel(x, y, walkable[x, y] ? floorColor : wallColor);
        mazeTexture.SetPixel(startCell.x, startCell.y, startColor);
        mazeTexture.SetPixel(goalCell.x, goalCell.y, goalColor);
        mazeTexture.Apply(false, false);

        mapRenderer.sprite = Sprite.Create(mazeTexture, new Rect(0f, 0f, mazeWidth, mazeHeight), new Vector2(0.5f, 0.5f), 1f);
        mapRenderer.transform.position = new Vector3((mazeWidth - 1) * 0.5f, (mazeHeight - 1) * 0.5f, 0f);
        playerRenderer.transform.position = GridToWorld(playerCell);
    }

    private void ConfigureCamera()
    {
        gameCamera = Camera.main;
        if (gameCamera == null) return;
        gameCamera.orthographic = true;
        gameCamera.backgroundColor = wallColor;
        UpdateCameraPosition();
    }

    private void ToggleCameraView()
    {
        followPlayerCamera = !followPlayerCamera;
        UpdateCameraPosition();
        status = followPlayerCamera ? "Player-follow camera enabled." : "Full-map overview enabled.";
    }

    private void UpdateCameraPosition()
    {
        if (gameCamera == null) gameCamera = Camera.main;
        if (gameCamera == null) return;

        float requiredByHeight = mazeHeight * 0.5f + 1.5f;
        float requiredByWidth = mazeWidth / (2f * Mathf.Max(gameCamera.aspect, 0.1f)) + 1.5f;
        if (followPlayerCamera)
        {
            gameCamera.orthographicSize = playerViewSize;
            Vector3 playerPosition = GridToWorld(playerCell);
            gameCamera.transform.position = new Vector3(playerPosition.x, playerPosition.y, -10f);
            return;
        }

        gameCamera.orthographicSize = Mathf.Max(requiredByHeight, requiredByWidth);
        gameCamera.transform.position = new Vector3((mazeWidth - 1) * 0.5f, (mazeHeight - 1) * 0.5f, -10f);
    }

    private Vector3 GridToWorld(Vector2Int cell) => new Vector3(cell.x, cell.y, 0f);
    private bool IsInterior(Vector2Int cell) => cell.x > 0 && cell.x < mazeWidth - 1 && cell.y > 0 && cell.y < mazeHeight - 1;
    private bool IsWalkable(Vector2Int cell) => cell.x >= 0 && cell.x < mazeWidth && cell.y >= 0 && cell.y < mazeHeight && walkable[cell.x, cell.y];
    private static int MakeOdd(int value) => value % 2 == 0 ? value + 1 : value;

    private static bool TryParseDirection(string raw, out Vector2Int direction)
    {
        string command = (raw ?? string.Empty).Trim().ToLowerInvariant();
        switch (command)
        {
            case "up": case "north": case "forward": case "w": case "move_up": direction = Vector2Int.up; return true;
            case "down": case "south": case "back": case "backward": case "s": case "move_down": direction = Vector2Int.down; return true;
            case "left": case "west": case "a": case "move_left": direction = Vector2Int.left; return true;
            case "right": case "east": case "d": case "move_right": direction = Vector2Int.right; return true;
            default: direction = Vector2Int.zero; return false;
        }
    }

    private void OnGUI()
    {
        if (walkable == null) return;
        const int width = 440;
        GUI.Box(new Rect(16, 16, width, 136), "BCI Maze — Large Random Map");
        GUI.Label(new Rect(30, 45, width - 24, 24), "Move: WASD / Arrow keys  |  New maze: R  |  Hint: H  |  View: V");
        GUI.Label(new Rect(30, 68, width - 24, 24), "External UDP: 127.0.0.1:7777  (up, down, left, right)");
        GUI.Label(new Rect(30, 91, width - 24, 24), status);
        if (ExternalMazeInput.Instance != null)
            GUI.Label(new Rect(30, 114, width - 24, 24), ExternalMazeInput.Instance.LinkStatus);
    }
}
