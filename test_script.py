from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator(r'log/Mon Jul  6 22-51-32 2020/events.out.tfevents.1594047092.Zhao-Jingwei')
ea.Reload()
print(ea.scalars.Keys())

val_acc=ea.scalars.Items('val/loss_total-epoch')
print(len(val_acc), val_acc[-1].value)