def streaming_collate_fn(batch):
    batch = batch[0]
    batch['image'] = batch['image'].unsqueeze(0)
    batch['mask'] = batch['mask'].unsqueeze(0) if batch['mask'] is not None else None
    batch['label'] = batch['label'].unsqueeze(0)
    return batch
