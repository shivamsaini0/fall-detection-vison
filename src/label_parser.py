"""Parse annotation formats from different fall detection datasets."""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VideoAnnotation:
    """Annotation for a single video file."""

    video_path: Path
    label: str  # "fall" or "adl" (activities of daily living / normal)
    fall_start_frame: int | None = None  # Frame where fall begins
    fall_end_frame: int | None = None  # Frame where fall ends
    dataset_name: str = ""

    @property
    def is_fall(self) -> bool:
        return self.label == "fall"


def parse_ur_fall_dataset(dataset_dir: str | Path) -> list[VideoAnnotation]:
    """Parse the UR Fall Detection Dataset.

    Supports two formats:
    1. Image sequence folders (Kaggle download):
        dataset_dir/
            UR_fall_detection_dataset_cam0_rgb/
                fall-01-cam0-rgb/   (folder of PNGs)
                adl-01-cam0-rgb/    (folder of PNGs)
    2. Video files:
        dataset_dir/
            fall-01-cam0-rgb.avi
            adl-01-cam0-rgb.avi

    Fall sequences are prefixed with 'fall-', ADL sequences with 'adl-'.

    Args:
        dataset_dir: Path to the UR Fall dataset directory.

    Returns:
        List of VideoAnnotation objects.
    """
    dataset_dir = Path(dataset_dir)
    annotations = []

    if not dataset_dir.exists():
        logger.warning(f"UR Fall dataset directory not found: {dataset_dir}")
        return annotations

    # Check for nested Kaggle structure
    cam0_dir = dataset_dir / "UR_fall_detection_dataset_cam0_rgb"
    if cam0_dir.exists():
        scan_dir = cam0_dir
    else:
        scan_dir = dataset_dir

    video_extensions = {".avi", ".mp4", ".mov"}

    for item in sorted(scan_dir.iterdir()):
        name = item.name.lower()

        if name.startswith("fall"):
            label = "fall"
        elif name.startswith("adl"):
            label = "adl"
        else:
            continue

        # Accept either directories (image sequences) or video files
        if item.is_dir():
            # Image sequence folder — check it has images
            image_files = [
                f for f in item.iterdir()
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            ]
            if not image_files:
                logger.warning(f"Empty image sequence, skipping: {item.name}")
                continue
            annotations.append(VideoAnnotation(
                video_path=item,
                label=label,
                dataset_name="ur_fall",
            ))
        elif item.suffix.lower() in video_extensions:
            annotations.append(VideoAnnotation(
                video_path=item,
                label=label,
                dataset_name="ur_fall",
            ))

    logger.info(
        f"UR Fall: {len(annotations)} sequences "
        f"({sum(1 for a in annotations if a.is_fall)} falls, "
        f"{sum(1 for a in annotations if not a.is_fall)} ADL)"
    )
    return annotations


def parse_le2i_dataset(dataset_dir: str | Path) -> list[VideoAnnotation]:
    """Parse the Le2i Fall Detection Dataset (IMVIA/Kaggle version).

    Supports the Kaggle structure where each room has a nested subfolder:
        dataset_dir/
            Coffee_room_01/
                Coffee_room_01/
                    Videos/
                        video (1).avi
                    Annotation_files/
                        video (1).txt

    Annotation format (per file):
        Line 1: fall_start_frame (int)
        Line 2: fall_end_frame (int)
        Remaining lines: frame_num,person_id,x1,y1,x2,y2

    If annotation has fall_start=0 and fall_end=0, it's a non-fall video.

    Args:
        dataset_dir: Path to the Le2i dataset directory.

    Returns:
        List of VideoAnnotation objects.
    """
    dataset_dir = Path(dataset_dir)
    annotations = []

    if not dataset_dir.exists():
        logger.warning(f"Le2i dataset directory not found: {dataset_dir}")
        return annotations

    # Find all room directories (possibly nested like Room/Room/)
    def find_videos_and_annots(base_dir: Path) -> tuple[Path | None, Path | None]:
        """Search for Videos/ and Annotation_files/ dirs, handling nesting."""
        video_exts = {".avi", ".mp4", ".mov"}

        # Check current dir for Videos/ subfolder
        videos_dir = base_dir / "Videos"
        annot_dir = base_dir / "Annotation_files"
        if videos_dir.exists():
            return videos_dir, annot_dir if annot_dir.exists() else None

        # Check one level deeper (Kaggle nesting: Room/Room/Videos/)
        for sub in sorted(base_dir.iterdir()):
            if sub.is_dir():
                videos_dir = sub / "Videos"
                annot_dir = sub / "Annotation_files"
                if videos_dir.exists():
                    return videos_dir, annot_dir if annot_dir.exists() else None

                # Some rooms have videos directly in nested folder (no Videos/ subdir)
                has_videos = any(
                    f.suffix.lower() in video_exts for f in sub.iterdir() if f.is_file()
                )
                if has_videos:
                    return sub, annot_dir if annot_dir.exists() else None

        return None, None

    for room_dir in sorted(dataset_dir.iterdir()):
        if not room_dir.is_dir():
            continue

        videos_dir, annot_dir = find_videos_and_annots(room_dir)
        if videos_dir is None:
            continue

        video_extensions = {".avi", ".mp4", ".mov"}
        for video_path in sorted(videos_dir.iterdir()):
            if video_path.suffix.lower() not in video_extensions:
                continue

            fall_start = None
            fall_end = None
            label = "adl"

            # Look for matching annotation file
            if annot_dir:
                annot_path = annot_dir / f"{video_path.stem}.txt"
                if annot_path.exists():
                    try:
                        lines = annot_path.read_text().strip().splitlines()
                        if len(lines) >= 2:
                            fs = int(lines[0].strip())
                            fe = int(lines[1].strip())
                            if fs > 0 and fe > 0:
                                fall_start = fs
                                fall_end = fe
                                label = "fall"
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse annotation {annot_path}: {e}")

            annotations.append(VideoAnnotation(
                video_path=video_path,
                label=label,
                fall_start_frame=fall_start,
                fall_end_frame=fall_end,
                dataset_name="le2i",
            ))

    logger.info(
        f"Le2i: {len(annotations)} videos "
        f"({sum(1 for a in annotations if a.is_fall)} falls, "
        f"{sum(1 for a in annotations if not a.is_fall)} ADL)"
    )
    return annotations


def parse_generic_folder(dataset_dir: str | Path) -> list[VideoAnnotation]:
    """Parse a generic dataset with fall/ and normal/ subdirectories.

    Expected structure:
        dataset_dir/
            fall/
                video1.mp4
                video2.mp4
            normal/  (or adl/)
                video3.mp4
                video4.mp4

    Args:
        dataset_dir: Path to the dataset directory.

    Returns:
        List of VideoAnnotation objects.
    """
    dataset_dir = Path(dataset_dir)
    annotations = []

    if not dataset_dir.exists():
        logger.warning(f"Dataset directory not found: {dataset_dir}")
        return annotations

    video_extensions = {".avi", ".mp4", ".mov", ".mkv"}
    label_dirs = {
        "fall": "fall",
        "falls": "fall",
        "normal": "adl",
        "adl": "adl",
        "no_fall": "adl",
    }

    for subdir in sorted(dataset_dir.iterdir()):
        if not subdir.is_dir():
            continue
        label = label_dirs.get(subdir.name.lower())
        if label is None:
            continue

        for video_path in sorted(subdir.iterdir()):
            if video_path.suffix.lower() in video_extensions:
                annotations.append(VideoAnnotation(
                    video_path=video_path,
                    label=label,
                    dataset_name="generic",
                ))

    logger.info(
        f"Generic: {len(annotations)} videos "
        f"({sum(1 for a in annotations if a.is_fall)} falls, "
        f"{sum(1 for a in annotations if not a.is_fall)} ADL)"
    )
    return annotations
