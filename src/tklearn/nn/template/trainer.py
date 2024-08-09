class TemplateBasedTrainer(Trainer):
    def fit(
        self,
        model: BaseModule,
        train_dataloader: Iterable[RecordBatch],
        val_dataloader: Iterable[RecordBatch],
        callbacks: Union[Iterable[Callback], CallbackList, None] = None,
    ):
        optimizer, lr_scheduler = model
        callbacks = CallbackList(callbacks)
        for epoch in range(10):
            for batch_idx, batch in enumerate(train_dataloader):
                # enable grads
                torch.set_grad_enabled(True)
                losses = []
                for batch in train_dataloader:
                    # calls hooks like this one
                    on_train_batch_start()
                    # train step
                    loss = training_step(batch)
                    # clear gradients
                    optimizer.zero_grad()
                    # backward
                    loss.backward()
                    # update parameters
                    optimizer.step()
                    losses.append(loss)
            for batch_idx, batch in enumerate(val_dataloader):
                model.validation_step(batch, batch_idx)


model: BaseModule = ...
train_dataloader: Iterable[RecordBatch] = ...
val_dataloader: Iterable[RecordBatch] = ...

trainer = Trainer()
trainer.fit(model, train_dataloader, val_dataloader)
