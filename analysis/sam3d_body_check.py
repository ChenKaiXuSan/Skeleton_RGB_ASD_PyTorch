from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
from pathlib import Path

try:
	import cv2
except Exception:  # pragma: no cover
	cv2 = None


DISEASES = ("ASD", "DHS", "LCS", "HipOA")


class Tee:
	def __init__(self, *streams) -> None:
		self.streams = streams

	def write(self, data: str) -> int:
		for stream in self.streams:
			stream.write(data)
		return len(data)

	def flush(self) -> None:
		for stream in self.streams:
			stream.flush()


def _add_item_name(item: Path, sample_names: set[str]) -> None:
	if item.name.startswith("."):
		return
	if item.is_dir():
		sample_names.add(item.name)
	elif item.is_file():
		sample_names.add(item.stem)


def list_sample_names(category_dir: Path) -> set[str]:
	if not category_dir.exists():
		return set()
	sample_names: set[str] = set()
	for item in category_dir.iterdir():
		if item.name.startswith("."):
			continue
		_add_item_name(item, sample_names)
	return sample_names


def count_files_recursive(root_dir: Path) -> int:
	if not root_dir.exists() or not root_dir.is_dir():
		return 0
	count = 0
	for path in root_dir.rglob("*"):
		if path.is_file() and not path.name.startswith("."):
			count += 1
	return count


def get_sample_data_dir(sample_dir: Path) -> Path:
	video_subdir = sample_dir / "video"
	if video_subdir.exists() and video_subdir.is_dir():
		return video_subdir
	return sample_dir


def get_result_frame_indices(sample_dir: Path) -> set[int]:
	data_dir = get_sample_data_dir(sample_dir)
	indices: set[int] = set()
	for path in data_dir.glob("*_sam3d_body.npz"):
		prefix = path.name.split("_", 1)[0]
		if prefix.isdigit():
			indices.add(int(prefix))
	return indices


def parse_none_detected_frames(sample_dir: Path) -> set[int]:
	none_file = get_sample_data_dir(sample_dir) / "none_detected_frames.txt"
	if not none_file.exists():
		return set()
	indices: set[int] = set()
	with none_file.open("r", encoding="utf-8") as file:
		for raw_line in file:
			line = raw_line.strip()
			if not line:
				continue
			if line.isdigit():
				indices.add(int(line))
	return indices


def get_video_frame_count(video_file: Path) -> int:
	if cv2 is not None:
		capture = cv2.VideoCapture(str(video_file))
		if capture.isOpened():
			frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
			capture.release()
			if frame_count > 0:
				return frame_count
		capture.release()

	cmd = [
		"ffprobe",
		"-v",
		"error",
		"-select_streams",
		"v:0",
		"-count_frames",
		"-show_entries",
		"stream=nb_read_frames",
		"-of",
		"default=nokey=1:noprint_wrappers=1",
		str(video_file),
	]
	result = subprocess.run(cmd, capture_output=True, text=True)
	if result.returncode != 0:
		raise RuntimeError(f"Failed to read frames with ffprobe: {video_file}")
	stdout = result.stdout.strip()
	if not stdout:
		raise RuntimeError(f"No frame count returned for video: {video_file}")
	return int(stdout)


def compare_categories(results_root: Path, video_root: Path) -> int:
	print("=" * 72)
	print(f"sam3d_body_results: {results_root}")
	print(f"video            : {video_root}")
	print("=" * 72)

	category_mismatch_count = 0
	frame_mismatch_count = 0
	none_txt_mismatch_count = 0
	missing_video_count = 0

	for disease in DISEASES:
		results_category = results_root / disease
		video_category = video_root / disease

		results_names = list_sample_names(results_category)
		video_names = list_sample_names(video_category)

		results_num = len(results_names)
		video_num = len(video_names)
		count_matched = results_num == video_num
		name_set_matched = results_names == video_names

		status = "OK" if count_matched else "MISMATCH"
		print(f"[{status}] {disease} count: results={results_num}, video={video_num}")

		if not count_matched:
			category_mismatch_count += 1

		if not name_set_matched:
			only_in_results = sorted(results_names - video_names)
			only_in_video = sorted(video_names - results_names)

			if only_in_results:
				print(f"  - only in sam3d_body_results ({len(only_in_results)}):")
				for name in only_in_results[:20]:
					print(f"    {name}")
				if len(only_in_results) > 20:
					print(f"    ... ({len(only_in_results) - 20} more)")

			if only_in_video:
				print(f"  - only in video ({len(only_in_video)}):")
				for name in only_in_video[:20]:
					print(f"    {name}")
				if len(only_in_video) > 20:
					print(f"    ... ({len(only_in_video) - 20} more)")

		common_names = sorted(results_names & video_names)
		disease_frame_mismatches = 0
		disease_none_mismatches = 0
		for sample_name in common_names:
			sample_dir = results_category / sample_name
			video_file = video_category / f"{sample_name}.mp4"

			if not video_file.exists():
				missing_video_count += 1
				continue

			result_indices = get_result_frame_indices(sample_dir)
			result_file_count = len(result_indices)
			video_frame_count = get_video_frame_count(video_file)

			if result_file_count != video_frame_count:
				disease_frame_mismatches += 1
				frame_mismatch_count += 1
				if disease_frame_mismatches <= 20:
					print(
						f"  - frame mismatch {sample_name}: "
						f"result_files={result_file_count}, video_frames={video_frame_count}"
					)

			expected_missing = set(range(video_frame_count)) - result_indices
			none_indices = parse_none_detected_frames(sample_dir)
			if expected_missing != none_indices:
				disease_none_mismatches += 1
				none_txt_mismatch_count += 1
				if disease_none_mismatches <= 20:
					print(
						f"  - none_detected mismatch {sample_name}: "
						f"expected_missing={len(expected_missing)}, recorded={len(none_indices)}"
					)
					only_expected = sorted(expected_missing - none_indices)
					only_recorded = sorted(none_indices - expected_missing)
					if only_expected:
						preview = ", ".join(str(x) for x in only_expected[:10])
						print(f"    missing in txt: {preview}{' ...' if len(only_expected) > 10 else ''}")
					if only_recorded:
						preview = ", ".join(str(x) for x in only_recorded[:10])
						print(f"    extra in txt  : {preview}{' ...' if len(only_recorded) > 10 else ''}")

		if disease_frame_mismatches > 20:
			print(f"  - ... ({disease_frame_mismatches - 20} more frame mismatches)")
		if disease_none_mismatches > 20:
			print(f"  - ... ({disease_none_mismatches - 20} more none_detected mismatches)")

	print("=" * 72)
	if (
		category_mismatch_count == 0
		and frame_mismatch_count == 0
		and none_txt_mismatch_count == 0
		and missing_video_count == 0
	):
		print(
			"All categories match: folder counts, frame counts, and none_detected records are consistent."
		)
		return 0

	print(
		"Summary: "
		f"category_count_mismatches={category_mismatch_count}, "
		f"frame_mismatches={frame_mismatch_count}, "
		f"none_txt_mismatches={none_txt_mismatch_count}, "
		f"missing_videos={missing_video_count}"
	)
	return 1


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Check whether sam3d_body_results and video have matching samples "
			"for ASD/DHS/LCS/HipOA."
		)
	)
	parser.add_argument(
		"--results-root",
		type=Path,
		default=Path("/work/SSR/share/data/asd_dataset/skeleton_rgb_dataset/sam3d_body_results"),
		help="Path to sam3d_body_results root folder.",
	)
	parser.add_argument(
		"--video-root",
		type=Path,
		default=Path("/work/SSR/share/data/asd_dataset/skeleton_rgb_dataset/video"),
		help="Path to video root folder.",
	)
	parser.add_argument(
		"--report-path",
		type=Path,
		default=Path(__file__).resolve().parent / "sam3d_body_check_report.txt",
		help="Path to save check report.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	args.report_path.parent.mkdir(parents=True, exist_ok=True)
	with args.report_path.open("w", encoding="utf-8") as report_file:
		tee = Tee(sys.stdout, report_file)
		with contextlib.redirect_stdout(tee):
			exit_code = compare_categories(args.results_root, args.video_root)
	print(f"Report saved to: {args.report_path}")
	raise SystemExit(exit_code)


if __name__ == "__main__":
	main()
