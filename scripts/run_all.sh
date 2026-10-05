#!/usr/bin/env bash
# Regenerate every reported output from raw inputs.  Usage: bash scripts/run_all.sh
set -e
D="data/raw/sample/Assignment - YC Startup"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
for c in c00a170fe1:single_room 1a8384c3f6:floor_only c7d28f72c6:with_ceiling; do
  id=${c%%:*}; name=${c##*:}
  roomscan run "$D/$id" --out out/lidar_$name
  roomscan run "$D/$id" --out out/lidar_${name}_nodrift --no-drift-correction --no-damage
  [ -f data/raw/derived/${name}_video.mp4 ] || python scripts/stray_to_video.py "$D/$id" data/raw/derived/${name}_video.mp4
  roomscan run data/raw/derived/${name}_video.mp4 --out out/video_$name
  [ -d data/raw/derived/${name}_photo ] || python -m scripts.stray_to_photos "$D/$id" data/raw/derived/${name}_photo
  roomscan run data/raw/derived/${name}_photo --tier photo --out out/photo_$name
done
# fix loop: damage detector before / after on the LiDAR tier
for c in c00a170fe1:single_room 1a8384c3f6:floor_only c7d28f72c6:with_ceiling; do
  id=${c%%:*}; name=${c##*:}
  ROOMSCAN_DAMAGE_PROFILE=before roomscan run "$D/$id" --out out/fixloop_before_$name
  ROOMSCAN_DAMAGE_PROFILE=after roomscan run "$D/$id" --out out/fixloop_after_$name
done
python scripts/report.py > docs/benchmark_results.md
