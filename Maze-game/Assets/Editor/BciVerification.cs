using System;
using System.IO;
using System.Threading.Tasks;
using MCPForUnity.Editor.Services;
using UnityEditor;
using UnityEngine;

/// <summary>Explicit command-line entry point for local BCI integration checks.</summary>
public static class BciVerification
{
    // Unity.exe -projectPath <project> -executeMethod BciVerification.Start
    // Invoked explicitly; opening the project does not start services or Play mode.
    public static async void Start()
    {
        try
        {
            var server = MCPServiceLocator.Server;
            if (!server.IsLocalHttpServerReachable() && !server.StartLocalHttpServer(quiet: true))
                throw new InvalidOperationException("Could not launch the configured local Unity MCP server.");
            for (int attempt = 0; attempt < 60 && !server.IsLocalHttpServerReachable(); attempt++)
                await Task.Delay(500);
            if (!server.IsLocalHttpServerReachable())
                throw new TimeoutException("Local Unity MCP server did not become reachable.");
            if (!await MCPServiceLocator.Bridge.StartAsync())
                throw new InvalidOperationException("Unity MCP bridge did not connect.");
            File.WriteAllText("Library/bci-verification-start.txt", "connected");
            Debug.Log("BCI verification: Unity MCP connected. Play mode remains under explicit control.");
        }
        catch (Exception error)
        {
            File.WriteAllText("Library/bci-verification-start.txt", error.ToString());
            Debug.LogException(error);
        }
    }
}
