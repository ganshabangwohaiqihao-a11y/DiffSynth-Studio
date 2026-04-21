import os, torch, json
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    writer = None
    tensorboard_log_interval = 1
    if args is not None:
        tensorboard_log_interval = max(1, args.tensorboard_log_interval)
        if args.tensorboard_log_dir is not None and accelerator.is_main_process:
            if SummaryWriter is None:
                print("TensorBoard is disabled because torch.utils.tensorboard is not available.")
            else:
                os.makedirs(args.tensorboard_log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create learning rate scheduler based on args
    scheduler = None
    if args is not None and hasattr(args, 'lr_scheduler_type'):
        scheduler_type = args.lr_scheduler_type.lower()
        scheduler_kwargs = {}
        if hasattr(args, 'lr_scheduler_kwargs'):
            try:
                scheduler_kwargs = json.loads(args.lr_scheduler_kwargs)
            except (json.JSONDecodeError, ValueError):
                scheduler_kwargs = {}
        
        if scheduler_type == "cosine":
            # Cosine annealing: decay from learning_rate to eta_min over T_max steps
            T_max = scheduler_kwargs.get("T_max", num_epochs)
            eta_min = scheduler_kwargs.get("eta_min", learning_rate / 10)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == "linear":
            # Linear decay: linearly decay learning rate from initial to 0
            total_steps = scheduler_kwargs.get("total_steps", num_epochs * 1000)  # estimate if not provided
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, total_iters=total_steps)
        else:  # constant
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                if writer is not None and model_logger.num_steps % tensorboard_log_interval == 0:
                    writer.add_scalar("train/loss", loss.detach().float().item(), model_logger.num_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)
    if writer is not None:
        writer.flush()
        writer.close()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
