// Execute in Unity Play mode through the existing MCP execute_code entry point.
var game = MazeGameController.Instance;
if (game == null) throw new System.Exception("MazeGameController is not running");
int passed = 0;
System.Action<bool, string> check = (condition, name) => {
    if (!condition) throw new System.Exception(name);
    passed++;
};
long generation = game.MazeGeneration;
var cell = game.PlayerCell;
int heading = game.Heading;
int steps = game.Steps;
bool paused = game.Paused;
string state = game.State.ToString();
check(game.MatchesExternalState(generation, cell.x, cell.y, heading, steps, paused, state), "current state matches");
check(!game.MatchesExternalState(generation - 1, cell.x, cell.y, heading, steps, paused, state), "old generation rejected");
check(!game.MatchesExternalState(generation, cell.x, cell.y, heading, steps + 1, paused, state), "wrong visit rejected");
check(!game.MatchesExternalState(generation, cell.x, cell.y, heading, steps, !paused, state), "changed pause rejected");
check(!game.MatchesExternalState(generation, cell.x, cell.y, (heading + 1) % 4, steps, paused, state), "changed heading rejected");
game.GenerateMaze();
check(game.MazeGeneration == generation + 1, "restart advances generation");
check(!game.MatchesExternalState(generation, game.PlayerCell.x, game.PlayerCell.y, game.Heading, game.Steps, game.Paused, game.State.ToString()), "restart rejects old generation at identical coordinates");
game.TryMove("pause");
return new { passed, generation = game.MazeGeneration };
