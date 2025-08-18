# Single render: base model (energy), yesterday’s sampler/FX
$Seed = 4242
python ui/cli.py --only base `
  --keyword "mellow acoustic guitar, relaxed tempo, soft dynamics" `
  --duration 6 --highpass 120 --lowpass 10000 --peak 0.90 `
  --temperature 0.95 --top-k 150 --top-p 0.95 --cfg 1.6 --seed $Seed
