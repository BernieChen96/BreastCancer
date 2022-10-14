# -*- coding: utf-8 -*-
# @Time    : 2022/10/14 18:01
# @Author  : CMM
# @File    : ddpm_images.py
# @Software: PyCharm
import torch
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets

import config
from ddpm.ddpm import script_utils

if __name__ == '__main__':
    parser = config.get_ddpm_argument()
    args = parser.parse_args()
    config.post_ddpm_config(args)
    device = args.device
    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            run = wandb.init(
                project=args.project_name,
                entity='mingmingchen96',
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion)

        batch_size = args.batch_size

        train_dataset = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=script_utils.get_transform(),
        )

        test_dataset = datasets.CIFAR10(
            root=args.data_path,
            train=False,
            download=True,
            transform=script_utils.get_transform(),
        )

        train_loader = script_utils.cycle(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        ))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=0)

        acc_train_loss = 0

        for iteration in range(1, args.iterations + 1):
            diffusion.train()

            x, y = next(train_loader)
            x = x.to(device)
            y = y.to(device)

            if args.use_labels:
                loss = diffusion(x, y)
            else:
                loss = diffusion(x)

            acc_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.update_ema()

            if iteration % args.log_rate == 0:
                test_loss = 0
                with torch.no_grad():
                    diffusion.eval()
                    for x, y in test_loader:
                        x = x.to(device)
                        y = y.to(device)

                        if args.use_labels:
                            loss = diffusion(x, y)
                        else:
                            loss = diffusion(x)

                        test_loss += loss.item()

                if args.use_labels:
                    samples = diffusion.sample(10, device, y=torch.arange(10, device=device))
                else:
                    samples = diffusion.sample(10, device)

                samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()

                test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate

                print('[%d/%d] test_loss: %.4f train_loss: %.4f'
                      % (iteration, args.iterations,
                         test_loss, acc_train_loss))
                wandb.log({
                    "test_loss": test_loss,
                    "train_loss": acc_train_loss,
                    "samples": [wandb.Image(sample) for sample in samples],
                })

                acc_train_loss = 0

            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)

        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        if args.log_to_wandb:
            run.finish()
        print("Keyboard interrupt, run finished early")
