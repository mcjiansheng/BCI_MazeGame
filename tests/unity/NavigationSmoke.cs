// Executed as a method body through MCP execute_code while the demo is in Play mode.
var g = MazeGameController.Instance;
var checks = new System.Collections.Generic.List<string>();
System.Action<bool, string> check = (ok, name) => {
    if (!ok) throw new System.Exception("Navigation check failed: " + name);
    checks.Add(name);
};
int oldSeed = g.fixedSeed;
bool oldFixed = g.useFixedSeed;
float oldWait = g.junctionWaitSeconds;
bool oldAuto = g.autoUTurnAtDeadEnd;
g.enabled = false;
try
{
    g.useFixedSeed = true;
    g.fixedSeed = 42;
    g.GenerateMaze();
    var camera = UnityEngine.Camera.main;
    var before = camera.transform.position;
    g.TickNavigation(0.2f);
    check(UnityEngine.Vector3.Distance(before, camera.transform.position) > 0.1f, "automatic continuous movement");
    check(g.Steps == 0, "no grid teleport before arrival");
    g.TryMove("pause");
    before = camera.transform.position;
    g.TickNavigation(2);
    check(camera.transform.position == before, "pause freezes movement");
    g.TryMove("left");
    g.TryMove("right");
    check(g.PendingTurn == "right", "latest buffered choice wins");
    g.TryMove("cancel");
    check(g.PendingTurn == "", "cancel buffered choice");
    g.TryMove("resume");
    int heading = g.Heading;
    g.TryMove("uturn");
    for (int i = 0; i < 30 && g.State != MazeGameController.TravelState.Turning; i++) g.TickNavigation(0.1f);
    check(g.Heading == (heading + 2) % 4, "mid-corridor U-turn waits for cell center");
    g.TickNavigation(1);
    check(UnityEngine.Quaternion.Angle(camera.transform.rotation, UnityEngine.Quaternion.Euler(0, g.Heading * 90, 0)) < 0.1f, "smooth turn finishes at exact heading");

    int left = 0, right = 0, straight = 0, tee = 0;
    foreach (int seed in new[] { 1, 2, 3, 4, 5, 42, 20260920 })
    {
        g.fixedSeed = seed;
        g.GenerateMaze();
        int shortest = g.GetHintPath().Count - 1;
        for (int i = 0; i < 20000 && g.State != MazeGameController.TravelState.Complete; i++)
        {
            if (g.State == MazeGameController.TravelState.Junction)
            {
                var route = g.GetHintPath();
                var delta = route[1] - route[0];
                int desired = System.Array.IndexOf(MazeGameController.Directions, delta);
                int relative = (desired - g.Heading + 4) % 4;
                if (!g.CanForward)
                {
                    var cell = g.PlayerCell;
                    g.TickNavigation(20);
                    if (g.PlayerCell != cell || g.State != MazeGameController.TravelState.Junction)
                        throw new System.Exception("T-junction advanced without a choice");
                    tee++;
                }
                if (relative == 3) { g.TryMove("left"); left++; }
                else if (relative == 1) { g.TryMove("right"); right++; }
                else if (relative == 0) { g.TickNavigation(g.junctionWaitSeconds + 0.1f); straight++; }
                else throw new System.Exception("Unexpected reverse on shortest path");
            }
            g.TickNavigation(0.25f);
            if (!g.IsWalkable(g.PlayerCell)) throw new System.Exception("Player entered a wall");
        }
        check(g.State == MazeGameController.TravelState.Complete && g.PlayerCell == g.GoalCell, "reach exit, seed " + seed);
        check(g.Steps == shortest, "shortest path with relative decisions, seed " + seed);
        check(!g.TryMove("left"), "completion stops navigation, seed " + seed);
    }
    check(left > 0 && right > 0 && tee > 0, "both relative choices and T-junction coverage");

    // Face a real branch from a different approach to exercise its straight exit.
    g.GenerateMaze();
    var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
    bool foundBranch = false;
    for (int x = 1; x < g.mazeWidth - 1 && !foundBranch; x++)
    for (int y = 1; y < g.mazeHeight - 1 && !foundBranch; y++)
    {
        var cell = new UnityEngine.Vector2Int(x, y);
        if (!g.IsWalkable(cell)) continue;
        for (int direction = 0; direction < 4 && !foundBranch; direction++)
        {
            if (!g.IsWalkable(cell + MazeGameController.Directions[direction])
                || !g.IsWalkable(cell + MazeGameController.Directions[(direction + 2) % 4])
                || !(g.IsWalkable(cell + MazeGameController.Directions[(direction + 1) % 4])
                     || g.IsWalkable(cell + MazeGameController.Directions[(direction + 3) % 4]))) continue;
            typeof(MazeGameController).GetField("<PlayerCell>k__BackingField", flags).SetValue(g, cell);
            typeof(MazeGameController).GetField("<Heading>k__BackingField", flags).SetValue(g, direction);
            camera.transform.SetPositionAndRotation(g.CellToWorld(cell) + UnityEngine.Vector3.up * g.eyeHeight,
                UnityEngine.Quaternion.Euler(0, direction * 90, 0));
            typeof(MazeGameController).GetMethod("Arrive", flags).Invoke(g, null);
            check(g.State == MazeGameController.TravelState.Junction && g.CanForward, "branch with straight option");
            g.TickNavigation(0.1f);
            check(g.State == MazeGameController.TravelState.Junction, "branch grants decision time");
            g.TickNavigation(g.junctionWaitSeconds);
            check(g.State == MazeGameController.TravelState.Moving && g.Heading == direction, "countdown defaults to straight");
            straight++;
            foundBranch = true;
        }
    }
    check(foundBranch, "straight junction fixture found");

    // Reversing immediately returns to the start, a guaranteed dead end in a DFS maze.
    g.fixedSeed = 42;
    g.GenerateMaze();
    g.autoUTurnAtDeadEnd = false;
    g.TryMove("uturn");
    for (int i = 0; i < 100 && g.State != MazeGameController.TravelState.DeadEnd; i++) g.TickNavigation(0.2f);
    check(g.State == MazeGameController.TravelState.DeadEnd, "arrive at dead end");
    g.TickNavigation(10);
    check(g.State == MazeGameController.TravelState.DeadEnd, "optional manual dead-end wait");
    g.autoUTurnAtDeadEnd = true;
    heading = g.Heading;
    g.TickNavigation(0.1f);
    check(g.State == MazeGameController.TravelState.Turning && g.Heading == (heading + 2) % 4, "automatic dead-end recovery");
    g.GenerateMaze();
    check(g.Steps == 0 && g.Turns == 0 && g.PendingTurn == "" && !g.Paused, "restart resets navigation");
    check(!camera.orthographic, "perspective first-person camera");
    return new { passed = checks.Count, checks = checks, left = left, right = right, straight = straight, tee = tee };
}
finally
{
    g.fixedSeed = oldSeed;
    g.useFixedSeed = oldFixed;
    g.junctionWaitSeconds = oldWait;
    g.autoUTurnAtDeadEnd = oldAuto;
    g.GenerateMaze();
    g.TryMove("pause");
    g.enabled = true;
}
