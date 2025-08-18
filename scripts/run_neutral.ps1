# Single render: neutral adapter, your preferred combo
$Seed = 4244
python ui/cli.py --only neutral --adapter-strength 0.01 `
  --keyword "neutral piano arpeggio, slow tempo, no drums, soft reverb" `
  --duration 6 --highpass 120 --lowpass 10000 --peak 0.90 `
  --temperature 0.95 --top-k 150 --top-p 0.95 --cfg 1.6 --seed $Seed
