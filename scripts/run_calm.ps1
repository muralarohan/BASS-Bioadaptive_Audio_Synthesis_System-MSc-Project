$Seed = 4243
python ui/cli.py --only calm --adapter-strength 0.005 `
  --keyword "calm ambient pads, slow tempo, no drums, soft reverb" `
  --duration 6 --highpass 120 --lowpass 9500 --peak 0.90 `
  --temperature 0.95 --top-k 150 --top-p 0.95 --cfg 1.6 --seed $Seed
