using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>Procedural courtyard architecture with shared runtime materials.</summary>
public sealed class MazeWorldRenderer : MonoBehaviour
{
    private GameObject root;
    private readonly List<Material> materials = new List<Material>();
    public void Build(MazeGameController maze)
    {
        if (root != null) { root.SetActive(false); Destroy(root); }
        foreach (var material in materials) Destroy(material);
        materials.Clear();
        root = new GameObject("First Person Maze");
        root.transform.SetParent(transform, false);
        var wall = MakeMaterial("Porcelain concrete", new Color(0.77f, 0.82f, 0.81f));
        var cap = MakeMaterial("Wall coping", new Color(0.36f, 0.45f, 0.46f));
        var baseMat = MakeMaterial("Graphite foundation", new Color(0.16f, 0.22f, 0.24f));
        var floor = MakeMaterial("Walkway stone", new Color(0.46f, 0.54f, 0.55f));
        var tile = MakeMaterial("Walkway alternate", new Color(0.51f, 0.59f, 0.59f));
        var teal = MakeMaterial("Teal navigation strip", new Color(0.06f, 0.55f, 0.48f), true);
        var orange = MakeMaterial("Exit amber", new Color(1f, 0.45f, 0.12f), true);
        float s = maze.cellSize;
        const float height = 3.15f;
        Box("Foundation", new Vector3((maze.mazeWidth - 1) * s / 2, -0.18f, (maze.mazeHeight - 1) * s / 2),
            new Vector3(maze.mazeWidth * s, 0.3f, maze.mazeHeight * s), baseMat);
        for (int x = 0; x < maze.mazeWidth; x++)
        for (int y = 0; y < maze.mazeHeight; y++)
        {
            var cell = new Vector2Int(x, y);
            var position = maze.CellToWorld(cell);
            if (maze.IsWalkable(cell))
            {
                Box("Paving", position + Vector3.down * 0.035f, new Vector3(s - 0.025f, 0.08f, s - 0.025f),
                    (x + y) % 2 == 0 ? floor : tile);
                // Edge strips show geometry, not the solution route.
                foreach (var direction in MazeGameController.Directions)
                {
                    if (maze.IsWalkable(cell + direction)) continue;
                    Vector3 offset = new Vector3(direction.x, 0, direction.y) * (s * 0.5f - 0.035f);
                    Box("Guide strip", position + offset + Vector3.up * 0.13f,
                        direction.x == 0 ? new Vector3(s, 0.055f, 0.035f) : new Vector3(0.035f, 0.055f, s), teal);
                }
                continue;
            }
            Box("Wall", position + Vector3.up * (height * 0.5f), new Vector3(s, height, s), wall);
            Box("Plinth", position + Vector3.up * 0.15f, new Vector3(s + 0.01f, 0.3f, s + 0.01f), baseMat);
            Box("Coping", position + Vector3.up * (height + 0.06f), new Vector3(s + 0.08f, 0.12f, s + 0.08f), cap);
        }
        var start = maze.CellToWorld(new Vector2Int(1, 1));
        Box("Start tile", start + Vector3.up * 0.015f, new Vector3(s * 0.65f, 0.025f, s * 0.65f), teal);
        var goal = maze.CellToWorld(maze.GoalCell);
        Box("Exit tile", goal + Vector3.up * 0.025f, new Vector3(s * 0.8f, 0.035f, s * 0.8f), orange);
        Box("Exit beacon", goal + Vector3.up * 4.5f, new Vector3(0.18f, 3.5f, 0.18f), orange);
        foreach (var direction in MazeGameController.Directions)
        {
            if (!maze.IsWalkable(maze.GoalCell + direction)) continue;
            var gate = new GameObject("Exit portal");
            gate.transform.SetParent(root.transform);
            gate.transform.position = goal;
            gate.transform.rotation = Quaternion.LookRotation(new Vector3(direction.x, 0, direction.y));
            GatePart(gate.transform, new Vector3(-s * 0.38f, 1.35f, 0), new Vector3(0.12f, 2.7f, 0.16f), orange);
            GatePart(gate.transform, new Vector3(s * 0.38f, 1.35f, 0), new Vector3(0.12f, 2.7f, 0.16f), orange);
            GatePart(gate.transform, new Vector3(0, 2.65f, 0), new Vector3(s * 0.8f, 0.15f, 0.16f), orange);
        }
        var sun = new GameObject("Courtyard sun");
        sun.transform.SetParent(root.transform);
        sun.transform.rotation = Quaternion.Euler(48, -32, 0);
        var light = sun.AddComponent<Light>();
        light.type = LightType.Directional;
        light.color = new Color(1f, 0.96f, 0.87f);
        light.intensity = 1.1f;
        light.shadows = LightShadows.Soft;
        RenderSettings.ambientMode = AmbientMode.Trilight;
        RenderSettings.ambientSkyColor = new Color(0.68f, 0.79f, 0.84f);
        RenderSettings.ambientEquatorColor = new Color(0.50f, 0.57f, 0.58f);
        RenderSettings.ambientGroundColor = new Color(0.28f, 0.31f, 0.32f);
        RenderSettings.fog = true;
        RenderSettings.fogMode = FogMode.Linear;
        RenderSettings.fogColor = new Color(0.66f, 0.82f, 0.89f);
        RenderSettings.fogStartDistance = 45;
        RenderSettings.fogEndDistance = 110;
    }
    private Material MakeMaterial(string name, Color color, bool emission = false)
    {
        var material = new Material(Shader.Find("Standard")) { name = name, color = color };
        material.SetFloat("_Glossiness", 0.18f);
        if (emission) { material.EnableKeyword("_EMISSION"); material.SetColor("_EmissionColor", color * 0.45f); }
        materials.Add(material);
        return material;
    }
    private GameObject Box(string name, Vector3 position, Vector3 scale, Material material)
    {
        var box = GameObject.CreatePrimitive(PrimitiveType.Cube);
        box.name = name;
        box.transform.SetParent(root.transform, false);
        box.transform.position = position;
        box.transform.localScale = scale;
        box.GetComponent<Renderer>().sharedMaterial = material;
        // The grid controls movement; this demo does not need a physics controller.
        Destroy(box.GetComponent<Collider>());
        return box;
    }
    private void GatePart(Transform gate, Vector3 position, Vector3 size, Material material)
    {
        var part = Box("Portal frame", Vector3.zero, size, material);
        part.transform.SetParent(gate, false);
        part.transform.localPosition = position;
    }
    private void OnDestroy()
    {
        foreach (var material in materials) if (material != null) Destroy(material);
    }
}
