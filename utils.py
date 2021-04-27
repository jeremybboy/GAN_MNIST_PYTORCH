import torchvision


def plot_loss_dicos(writer, epoch_id, monitored_quantities_train,
                    monitored_quantities_test):

    for k, v in monitored_quantities_train.items():
        if type(v) == list:
            for ind, elem in enumerate(v):
                writer.add_scalar(f'{k}_{ind}/train', elem, epoch_id)
        else:
            writer.add_scalar(f'{k}/train', v, epoch_id)

    for k, v in monitored_quantities_test.items():
        if type(v) == list:
            for ind, elem in enumerate(v):
                writer.add_scalar(f'{k}_{ind}/test', elem, epoch_id)
        else:
            writer.add_scalar(f'{k}/test', v, epoch_id)


def plot_examples(writer, epoch_id, image_batch):
    n_row = image_batch.size(0) // 4
    grid = torchvision.utils.make_grid(image_batch, nrow=n_row)
    writer.add_image('generations', grid, epoch_id)
