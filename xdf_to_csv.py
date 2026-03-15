from __future__ import annotations

import argparse

from igaze.xdf import convert_subjects_from_config, convert_xdf_to_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert XDF streams to CSV files.")
    parser.add_argument("xdf_path", nargs="?", help="Path to the input XDF file.")
    parser.add_argument("--config", help="Config file containing eyetracking subject XDF paths.")
    parser.add_argument(
        "--output-dir",
        help="Directory where CSV files will be written.",
    )
    args = parser.parse_args()

    if args.config:
        results = convert_subjects_from_config(args.config, args.output_dir)
        total_files = sum(len(paths) for paths in results.values())
        print(f"Wrote {total_files} CSV file(s) for {len(results)} subject(s):")
        for subject_id, paths in results.items():
            print(f"subject {subject_id}:")
            for file_path in paths:
                print(file_path)
        return

    if not args.xdf_path:
        parser.error("Provide either xdf_path or --config.")

    written_files = convert_xdf_to_csv(args.xdf_path, args.output_dir)
    print(f"Wrote {len(written_files)} CSV file(s):")
    for file_path in written_files:
        print(file_path)


if __name__ == "__main__":
    main()