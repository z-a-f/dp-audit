1. Print a configuration to have as reference
`python main.py fit --print_config > config.yaml`

2. Modify the config to your liking - you can remove all default arguments
`nano config.yaml`

3. Fit your model using the edited configuration
`python main.py fit --config config.yaml`