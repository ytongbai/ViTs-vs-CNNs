from torch.optim.lr_scheduler import _LRScheduler


def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None):
    global state
    if args.lr_schedule == 'cos':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        state['lr'] = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif args.lr_schedule == 'step':
        if epoch in args.schedule:
            state['lr'] *= args.gamma
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1, start_lr=0.1):
        self.total_iters = total_iters
        self.start_lr = start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        ret = [self.start_lr + (base_lr - self.start_lr) * self.last_epoch / (self.total_iters + 1e-8) for base_lr in
               self.base_lrs]
        return ret
