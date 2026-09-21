"""Visualization utilities for visible-image bbox inspection."""

from __future__ import annotations

import textwrap
import subprocess
import tempfile
from pathlib import Path

from .metrics import is_valid_bbox


def normalized_to_pixel_bbox(bbox, width: int, height: int):
    """Convert normalized [x1, y1, x2, y2] to integer pixel coordinates."""
    if not is_valid_bbox(bbox):
        raise ValueError(f"Invalid normalized bbox: {bbox}")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def draw_bbox_on_visible(sample, output_dir: str) -> str:
    """Draw the ground-truth bbox and query on the visible RGB image."""
    image_path = Path(sample["visible_path"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / f"{sample['sample_id']}_visible_bbox.jpg"

    ps_script = r"""
param(
    [string]$imagePath,
    [string]$savePath,
    [double]$x1n,
    [double]$y1n,
    [double]$x2n,
    [double]$y2n,
    [string]$query
)
Add-Type -AssemblyName System.Drawing

$src = [System.Drawing.Image]::FromFile($imagePath)
$bmp = New-Object System.Drawing.Bitmap $src.Width, $src.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.DrawImage($src, 0, 0, $src.Width, $src.Height)

$x1 = [int][Math]::Round($x1n * $src.Width)
$y1 = [int][Math]::Round($y1n * $src.Height)
$x2 = [int][Math]::Round($x2n * $src.Width)
$y2 = [int][Math]::Round($y2n * $src.Height)
$pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::Red), 2
$g.DrawRectangle($pen, $x1, $y1, [Math]::Max(1, $x2 - $x1), [Math]::Max(1, $y2 - $y1))

$font = New-Object System.Drawing.Font "Arial", 12
$white = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::White)
$black = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::Black)
$y = 8
foreach ($line in ($query -split "\n")) {
    $g.DrawString($line, $font, $black, 9, ($y + 1))
    $g.DrawString($line, $font, $white, 8, $y)
    $y += 18
}

$bmp.Save($savePath, [System.Drawing.Imaging.ImageFormat]::Jpeg)
$g.Dispose()
$src.Dispose()
$bmp.Dispose()
"""

    query = "\n".join(textwrap.wrap(str(sample.get("query", "")), width=48)[:3])
    bbox = sample["bbox"]
    if not is_valid_bbox(bbox):
        raise ValueError(f"Invalid normalized bbox: {bbox}")

    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8") as f:
        f.write(ps_script)
        ps_file = f.name

    try:
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                ps_file,
                str(image_path),
                str(save_path),
                str(float(bbox[0])),
                str(float(bbox[1])),
                str(float(bbox[2])),
                str(float(bbox[3])),
                query,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"Failed to save visualization: {save_path}\n{detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to save visualization: {save_path}") from exc
    finally:
        Path(ps_file).unlink(missing_ok=True)
    return str(save_path)
