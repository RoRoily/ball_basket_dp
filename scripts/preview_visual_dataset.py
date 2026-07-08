# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export visual dataset previews as PNG contact sheets and GIFs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


def _episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = np.concatenate(([0], episode_ends[:-1]))
    return [(int(start), int(end)) for start, end in zip(starts, episode_ends)]


def _decode_string(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _sample_indices(start: int, end: int, num_frames: int) -> np.ndarray:
    if end <= start:
        raise ValueError("Episode end must be greater than start.")
    count = min(num_frames, end - start)
    return np.linspace(start, end - 1, count, dtype=np.int64)


def _frame_image(frame: np.ndarray, scale: int) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    return image


def _make_contact_sheet(
    frames: np.ndarray,
    indices: np.ndarray,
    title: str,
    columns: int,
    scale: int,
) -> Image.Image:
    frame_images = [_frame_image(frame, scale) for frame in frames]
    frame_width, frame_height = frame_images[0].size
    label_height = 18
    title_height = 26
    columns = max(1, min(columns, len(frame_images)))
    rows = int(np.ceil(len(frame_images) / columns))
    sheet = Image.new("RGB", (columns * frame_width, title_height + rows * (frame_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), title, fill=(0, 0, 0))
    for item_index, image in enumerate(frame_images):
        row = item_index // columns
        column = item_index % columns
        x = column * frame_width
        y = title_height + row * (frame_height + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + frame_height + 2), f"t={int(indices[item_index])}", fill=(0, 0, 0))
    return sheet


def _save_gif(frames: np.ndarray, path: Path, fps: int, scale: int, max_gif_frames: int) -> None:
    if max_gif_frames > 0 and len(frames) > max_gif_frames:
        indices = np.linspace(0, len(frames) - 1, max_gif_frames, dtype=np.int64)
        frames = frames[indices]
    images = [_frame_image(frame, scale) for frame in frames]
    duration_ms = max(1, int(1000 / max(fps, 1)))
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)


def _optional_array(group: h5py.Group, name: str, default):
    return np.asarray(group[name]) if name in group else default


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a visual BallBasket HDF5 dataset.")
    parser.add_argument("dataset", type=str, help="Visual HDF5 dataset with data/images.")
    parser.add_argument("--output_dir", type=str, default=None, help="Preview output directory.")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Episode indices to preview.")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes if --episodes is omitted.")
    parser.add_argument("--frames_per_episode", type=int, default=12, help="Frames in each contact sheet.")
    parser.add_argument("--columns", type=int, default=6, help="Contact sheet columns.")
    parser.add_argument("--scale", type=int, default=2, help="Integer pixel scale for saved previews.")
    parser.add_argument("--gif", action="store_true", default=False, help="Also save per-episode GIFs.")
    parser.add_argument("--gif_fps", type=int, default=15, help="GIF playback FPS.")
    parser.add_argument("--max_gif_frames", type=int, default=160, help="Maximum GIF frames. 0 keeps all frames.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_stem = dataset_path.with_suffix("")
        output_dir = output_stem.parent / f"{output_stem.name}_preview"
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as h5_file:
        if "data/images" not in h5_file:
            raise ValueError("Dataset does not contain data/images. Use collect_visual_demos.py first.")
        data_group = h5_file["data"]
        images = data_group["images"]
        episode_ends = np.asarray(h5_file["meta/episode_ends"], dtype=np.int64)
        ranges = _episode_ranges(episode_ends)
        success = _optional_array(data_group, "success", np.zeros(len(ranges), dtype=np.bool_))
        attach_count = _optional_array(data_group, "attach_count", np.zeros(len(ranges), dtype=np.int64))
        plans = (
            [_decode_string(value) for value in data_group["plan"][:]]
            if "plan" in data_group
            else ["unknown"] * len(ranges)
        )
        final_distance = _optional_array(
            data_group,
            "final_ball_to_basket_distance",
            np.full(len(ranges), np.nan, dtype=np.float32),
        )

        if args.episodes:
            episode_indices = [int(index) for index in args.episodes]
        else:
            episode_indices = list(range(min(args.num_episodes, len(ranges))))
        invalid = [index for index in episode_indices if index < 0 or index >= len(ranges)]
        if invalid:
            raise ValueError(f"Invalid episode indices: {invalid}. Dataset has {len(ranges)} episodes.")

        manifest = {
            "dataset": str(dataset_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "num_episodes": len(ranges),
            "image_shape": tuple(int(value) for value in images.shape),
            "previews": [],
        }

        for episode_index in episode_indices:
            start, end = ranges[episode_index]
            sheet_indices = _sample_indices(start, end, args.frames_per_episode)
            sheet_frames = np.asarray(images[sheet_indices], dtype=np.uint8)
            title = (
                f"episode={episode_index} plan={plans[episode_index]} "
                f"success={bool(success[episode_index])} attaches={int(attach_count[episode_index])} "
                f"final_dist={float(final_distance[episode_index]):.3f}"
            )
            sheet = _make_contact_sheet(sheet_frames, sheet_indices - start, title, args.columns, args.scale)
            sheet_path = output_dir / f"episode_{episode_index:04d}_sheet.png"
            sheet.save(sheet_path)

            gif_path = None
            if args.gif:
                gif_path = output_dir / f"episode_{episode_index:04d}.gif"
                _save_gif(np.asarray(images[start:end], dtype=np.uint8), gif_path, args.gif_fps, args.scale, args.max_gif_frames)

            manifest["previews"].append(
                {
                    "episode": episode_index,
                    "start": start,
                    "end": end,
                    "length": end - start,
                    "plan": plans[episode_index],
                    "success": bool(success[episode_index]),
                    "attach_count": int(attach_count[episode_index]),
                    "final_ball_to_basket_distance": float(final_distance[episode_index]),
                    "sheet": str(sheet_path),
                    "gif": None if gif_path is None else str(gif_path),
                }
            )
            print(f"[INFO]: wrote contact sheet: {sheet_path}")
            if gif_path is not None:
                print(f"[INFO]: wrote GIF: {gif_path}")

    manifest_path = output_dir / "preview_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    print(f"[INFO]: wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
