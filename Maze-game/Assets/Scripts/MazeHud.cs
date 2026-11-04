using UnityEngine;

/// <summary>Compact demo HUD; mouse, keyboard and BCI share the same commands.</summary>
public sealed class MazeHud : MonoBehaviour
{
    private MazeGameController maze;
    private GUIStyle title, text, small, symbol, button;
    private bool settings;
    private readonly Color ink = new Color(0.035f, 0.085f, 0.095f, 0.92f);
    private readonly Color accent = new Color(0.32f, 0.95f, 0.78f);
    private void Awake() { maze = GetComponent<MazeGameController>(); }
    private void Styles()
    {
        if (title != null) return;
        title = new GUIStyle(GUI.skin.label) { fontSize = 22, fontStyle = FontStyle.Bold };
        text = new GUIStyle(GUI.skin.label) { fontSize = 16, alignment = TextAnchor.MiddleCenter };
        small = new GUIStyle(text) { fontSize = 12 };
        symbol = new GUIStyle(text) { fontSize = 30 };
        button = new GUIStyle(GUI.skin.button) { fontSize = 23, alignment = TextAnchor.MiddleCenter };
        title.normal.textColor = text.normal.textColor = small.normal.textColor = symbol.normal.textColor = Color.white;
    }
    private static void Fill(Rect rect, Color color)
    {
        Color previous = GUI.color;
        GUI.color = color;
        GUI.DrawTexture(rect, Texture2D.whiteTexture);
        GUI.color = previous;
    }
    private void OnGUI()
    {
        if (maze == null || maze.Minimap == null) return;
        Styles();
        float scale = Mathf.Min(1.5f, Mathf.Min(Screen.width / 960f, Screen.height / 640f));
        GUI.matrix = Matrix4x4.Scale(new Vector3(scale, scale, 1));
        float w = Screen.width / scale, h = Screen.height / scale;
        Fill(new Rect(20, 20, 252, 106), ink);
        GUI.Label(new Rect(36, 29, 230, 28), "NEURO / MAZE", title);
        GUI.Label(new Rect(36, 63, 220, 24), $"{maze.mazeWidth} × {maze.mazeHeight}     {maze.HeadingName}     {maze.Steps:000} steps", small);
        GUI.Label(new Rect(36, 88, 220, 23), ExternalMazeInput.Instance != null ? ExternalMazeInput.Instance.LinkStatus : "Keyboard mode", small);
        Fill(new Rect(w / 2 - 140, 20, 280, 42), ink);
        string state = maze.Paused ? "PAUSED" : maze.State == MazeGameController.TravelState.Moving ? "AUTO WALK"
            : maze.State == MazeGameController.TravelState.Turning ? "TURNING"
            : maze.State == MazeGameController.TravelState.Junction ? "CHOOSE A DIRECTION"
            : maze.State == MazeGameController.TravelState.DeadEnd ? "DEAD END" : "EXIT REACHED";
        GUI.Label(new Rect(w / 2 - 140, 20, 280, 42), state, text);
        if (maze.showMinimap)
        {
            const float mapSize = 160;
            var map = new Rect(w - mapSize - 36, 36, mapSize, mapSize);
            Fill(new Rect(map.x - 12, map.y - 12, mapSize + 24, mapSize + 47), ink);
            GUI.DrawTexture(map, maze.Minimap);
            float cellW = mapSize / maze.mazeWidth, cellH = mapSize / maze.mazeHeight;
            float px = map.x + (maze.PlayerCell.x + 0.5f) * cellW;
            float py = map.y + (maze.mazeHeight - maze.PlayerCell.y - 0.5f) * cellH;
            Fill(new Rect(px - 3, py - 3, 6, 6), Color.white);
            var direction = MazeGameController.Directions[maze.Heading];
            Fill(new Rect(px + direction.x * 6 - 2, py - direction.y * 6 - 2, 4, 4), accent);
            GUI.Label(new Rect(map.x, map.yMax + 7, mapSize, 24), "YOU  /  EXIT", small);
        }
        Fill(new Rect(w / 2 - 2, h / 2 - 2, 4, 4), new Color(1, 1, 1, 0.6f));
        if (maze.State == MazeGameController.TravelState.Complete)
        {
            Fill(new Rect(w / 2 - 210, h / 2 - 90, 420, 165), ink);
            GUI.Label(new Rect(w / 2 - 190, h / 2 - 74, 380, 40), "EXIT REACHED", symbol);
            GUI.Label(new Rect(w / 2 - 190, h / 2 - 24, 380, 26), $"{maze.Steps} steps   /   {maze.Turns} turns   /   {maze.ElapsedSeconds:F0}s", text);
            if (GUI.Button(new Rect(w / 2 - 95, h / 2 + 20, 190, 38), "New maze")) maze.GenerateMaze();
        }
        float bx = w / 2 - 162, by = h - 105;
        Fill(new Rect(bx - 12, by - 37, 348, 124), ink);
        string pending = maze.PendingTurn.Length > 0 ? "QUEUED: " + maze.PendingTurn.ToUpperInvariant() : "";
        string decision = maze.State == MazeGameController.TravelState.Junction && maze.CanForward
            ? $"AHEAD IN {maze.DecisionRemaining:F1}s" : maze.State == MazeGameController.TravelState.DeadEnd && maze.autoUTurnAtDeadEnd
            ? $"TURN AROUND IN {maze.DecisionRemaining:F1}s" : "";
        GUI.Label(new Rect(bx, by - 32, 324, 27), pending.Length > 0 ? pending : decision, small);
        TurnButton(new Rect(bx, by, 100, 48), "←", "Left", "left");
        TurnButton(new Rect(bx + 112, by, 100, 48), "↶", "Turn around", "uturn");
        TurnButton(new Rect(bx + 224, by, 100, 48), "→", "Right", "right");
        if (GUI.Button(new Rect(20, h - 62, 44, 42), new GUIContent(maze.Paused ? "▶" : "Ⅱ", "Pause / resume"), button)) maze.TogglePause();
        if (GUI.Button(new Rect(72, h - 62, 44, 42), new GUIContent("↻", "New maze"), button)) maze.GenerateMaze();
        if (GUI.Button(new Rect(124, h - 62, 44, 42), new GUIContent("≡", "Demo settings"), button)) settings = !settings;
        if (GUI.Button(new Rect(w - 64, h - 62, 44, 42), new GUIContent("?", "Route hint"), button)) maze.ShowHint();
        if (maze.Hint.Length > 0) GUI.Label(new Rect(w / 2 - 200, h - 170, 400, 24), maze.Hint, text);
        if (settings) DrawSettings(h);
        if (GUI.tooltip.Length > 0)
        {
            var tooltip = new Rect(20, h - 99, 180, 28);
            Fill(tooltip, ink);
            GUI.Label(tooltip, GUI.tooltip, small);
        }
        GUI.matrix = Matrix4x4.identity;
    }
    private void TurnButton(Rect rect, string icon, string label, string command)
    {
        GUI.backgroundColor = maze.PendingTurn == command ? accent : Color.white;
        GUI.enabled = maze.State != MazeGameController.TravelState.Complete;
        if (GUI.Button(rect, new GUIContent(icon, label), button)) maze.TryMove(command);
        GUI.enabled = true;
        GUI.backgroundColor = Color.white;
        GUI.Label(new Rect(rect.x, rect.yMax + 2, rect.width, 22), label, small);
    }
    private void DrawSettings(float height)
    {
        var panel = new Rect(20, height - 318, 280, 235);
        Fill(panel, ink);
        GUILayout.BeginArea(new Rect(panel.x + 16, panel.y + 12, panel.width - 32, panel.height - 24));
        GUILayout.Label("DEMO SETTINGS", title);
        GUILayout.Label($"Walking speed   {maze.moveSpeed:F1} m/s");
        maze.moveSpeed = GUILayout.HorizontalSlider(maze.moveSpeed, 1f, 5f);
        GUILayout.Label($"Junction wait   {maze.junctionWaitSeconds:F1} s");
        maze.junctionWaitSeconds = GUILayout.HorizontalSlider(maze.junctionWaitSeconds, 0f, 10f);
        maze.autoUTurnAtDeadEnd = GUILayout.Toggle(maze.autoUTurnAtDeadEnd, "Automatic dead-end U-turn");
        maze.showMinimap = GUILayout.Toggle(maze.showMinimap, "Minimap");
        if (GUILayout.Button("Cancel queued turn")) maze.TryMove("cancel");
        GUILayout.EndArea();
    }
}
