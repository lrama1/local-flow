import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { TutorialsPage } from "@/features/tutorials/TutorialsPage";
import { currentChapter, type TutorialManifest } from "@/features/tutorials/tutorials";
import { plural } from "@/lib/utils";
import { jsonResponse, renderWithProviders } from "./helpers";

const manifest: TutorialManifest = {
  version: 1,
  tutorials: [
    {
      id: "story-to-shots",
      title: "From story idea to shot list",
      summary: "Refine an idea and break it into shots.",
      duration_s: 163.6,
      video: "/tutorials/story-to-shots.mp4",
      poster: "/tutorials/story-to-shots.jpg",
      captions: "/tutorials/story-to-shots.vtt",
      chapters: [
        { time_s: 0.9, title: "Introduction" },
        { time_s: 36.5, title: "Refine the story" },
      ],
    },
    {
      id: "images-and-keyframes",
      title: "Reference images and keyframes",
      summary: "Generate references and compose a keyframe.",
      duration_s: 138.9,
      video: "/tutorials/images-and-keyframes.mp4",
      chapters: [],
    },
  ],
};

afterEach(() => vi.unstubAllGlobals());

function serve(body: Response) {
  const fetchMock = vi.fn(async (_url: string) => body);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

test("helpers: current chapter and plural", () => {
  const chapters = manifest.tutorials[0]!.chapters;
  expect(currentChapter(chapters, 0)).toBe(-1);
  expect(currentChapter(chapters, 10)).toBe(0);
  expect(currentChapter(chapters, 36.3)).toBe(1); // a seek lands a hair before the chapter start
  expect(plural(1, "scene")).toBe("1 scene");
  expect(plural(7, "shot")).toBe("7 shots");
});

test("plays the first tutorial with captions, seeks to chapters, and switches tutorials", async () => {
  const fetchMock = serve(jsonResponse(manifest));
  const play = vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
  renderWithProviders(<TutorialsPage />);

  const video = await screen.findByTestId("tutorial-video");
  expect(fetchMock.mock.calls[0]![0]).toBe("/tutorials/index.json");
  expect(video.querySelector("source")).toHaveAttribute("src", "/tutorials/story-to-shots.mp4");
  expect(video.querySelector("track")).toHaveAttribute("src", "/tutorials/story-to-shots.vtt");
  expect(screen.getByRole("heading", { name: "From story idea to shot list" })).toBeInTheDocument();

  const chapters = screen.getByRole("list", { name: "Chapters" });
  await userEvent.click(within(chapters).getByRole("button", { name: /Refine the story/ }));
  expect((video as HTMLVideoElement).currentTime).toBe(36.5);
  expect(play).toHaveBeenCalled();
  expect(within(chapters).getByRole("button", { name: /Refine the story/ })).toHaveAttribute(
    "aria-current",
    "true",
  );

  const list = screen.getByRole("list", { name: "Tutorial list" });
  await userEvent.click(within(list).getByRole("button", { name: /Reference images and keyframes/ }));
  await waitFor(() =>
    expect(screen.getByTestId("tutorial-video").querySelector("source")).toHaveAttribute(
      "src",
      "/tutorials/images-and-keyframes.mp4",
    ),
  );
  expect(screen.queryByRole("list", { name: "Chapters" })).not.toBeInTheDocument();
});

test("shows how to record tutorials when the manifest is missing", async () => {
  // The production server answers unknown paths with the SPA's index.html.
  serve(new Response("<!doctype html>", { status: 200, headers: { "Content-Type": "text/html" } }));
  renderWithProviders(<TutorialsPage />);
  expect(await screen.findByText("No tutorials yet")).toBeInTheDocument();
  expect(screen.getByText(/make_tutorials\.py/)).toBeInTheDocument();
});
