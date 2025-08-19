# Requires: venv activated and adapters present.
# Edit these two for your setup:
$Device = "24:AC:AC:06:E8:39"    # Polar Verity Sense address or name
$BaselineBpm = 75

# Session layout
$Segments = 4
$SegSec   = 20
$Xfade    = 1.5
$Seed     = 4241   # auto-increments per segment

python ui/live.py --device "$Device" `
  --baseline-bpm $BaselineBpm `
  --segments $Segments --segment-sec $SegSec --crossfade-sec $Xfade `
  --seed $Seed --log-level INFO
