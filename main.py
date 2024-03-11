import torch

import lightning.pytorch.cli as cli

def main():
    torch.set_float32_matmul_precision('high')  # Switch to 'medium' to speedup training
    main_cli = cli.LightningCLI()

if __name__ == '__main__':
    main()