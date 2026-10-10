import { createBrowserRouter, Navigate, type RouteObject } from "react-router";
import { CastLibraryPage } from "@/features/cast-library/CastLibraryPage";
import { ModelsPage } from "@/features/models/ModelsPage";
import { PlaygroundPage } from "@/features/playground/PlaygroundPage";
import { ProjectsPage } from "@/features/projects/ProjectsPage";
import { ProjectWorkspace } from "@/features/projects/ProjectWorkspace";
import { SettingsPage } from "@/features/settings/SettingsPage";
import { TutorialsPage } from "@/features/tutorials/TutorialsPage";
import { Layout } from "./Layout";

export const routes: RouteObject[] = [
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <Navigate to="/projects" replace /> },
      { path: "projects", element: <ProjectsPage /> },
      { path: "projects/:projectId/:tab?", element: <ProjectWorkspace /> },
      { path: "cast", element: <CastLibraryPage /> },
      { path: "playground", element: <PlaygroundPage /> },
      { path: "models", element: <ModelsPage /> },
      { path: "tutorials", element: <TutorialsPage /> },
      { path: "settings", element: <SettingsPage /> },
      { path: "*", element: <Navigate to="/projects" replace /> },
    ],
  },
];

export const router = createBrowserRouter(routes);
