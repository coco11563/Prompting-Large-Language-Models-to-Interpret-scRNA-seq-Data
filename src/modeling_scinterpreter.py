import os
from typing import Optional
from tqdm import tqdm

import torch
from torch import nn

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class netscInterpreter(nn.Module):
    def __init__(self,
                 pretrained_llm: Optional[str] = None,
                 num_classes: Optional[int] = None,
                 prompt: Optional[str] = None,
                 init_range: Optional[float] = 0.02,
                 ):
        super().__init__()
        if pretrained_llm is not None:
            from transformers import AutoModel, AutoTokenizer, LlamaModel
            self.llm = pretrained_llm.split("/")[-1]
            self.backbone = AutoModel.from_pretrained(pretrained_llm)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_llm)
            prompt_ids = self.tokenizer.tokenize(prompt) + [self.tokenizer.cls_token_id]
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
            self.register_buffer("prompt_embeds", self.backbone.embed_tokens(prompt_ids))
        else:
            self.backbone = None

        self.prompt = prompt

        for param in self.backbone.parameters():
            param.requires_grad = False

        gpt_embeddings = torch.load("./gpt_embeddings.bin", map_location="cpu")
        self.register_buffer("gpt_embeddings", gpt_embeddings)

        dllm = self.backbone.config.hidden_size if pretrained_llm is not None else 4096
        l1 = [layer for _ in range(4) for layer in [nn.SiLU(), nn.Linear(dllm, dllm)]]
        l1.insert(0, nn.Linear(gpt_embeddings.shape[1], dllm))
        self.l1 = nn.Sequential(*l1)

        l2 = [layer for _ in range(4) for layer in [nn.Linear(dllm, dllm), nn.SiLU()]]
        l2.append(nn.Linear(dllm, num_classes))
        self.l2 = nn.Sequential(*l2)

        self.num_classes = num_classes

        self.init_range = init_range
        self.l1.apply(self._init_weights), self.l2.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.init_range
        if hasattr(module, "weight"):
            module.weight.data.normal_(mean=0.0, std=std)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()

    def forward(self,
                input_ids: torch.LongTensor,
                values: torch.FloatTensor,
                attention_mask: Optional[torch.LongTensor],
                **kwargs,
                ):
        hidden_states: torch.Tensor = self.l1(self.gpt_embeddings[input_ids] * values.unsqueeze(-1))
        if self.backbone is not None:
            hidden_states = torch.cat([hidden_states, torch.stack([self.prompt_embeds for _ in range(hidden_states.shape[0])])], dim=1)
            hidden_states = self.backbone(
                input_embeds=hidden_states,
                attention_mask=attention_mask,
                **kwargs
                )[0][:,-1,:]
            logits = self.l2(hidden_states)
        else:
            hidden_states = hidden_states.mean(dim=1)
            logits = self.l2(hidden_states)

        return logits, hidden_states
    
    def fit(self,
            rank,
            dataset,
            manager,
            epochs=1,
            num_workers=0,
            num_gpus=1,
            loss_fn=nn.CrossEntropyLoss(),
            lr=1.,
            batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            log_wandb=False,
            log_prefix=None,
            display_metrics=None,
            **kwargs):

        from torch.utils.data import DataLoader, DistributedSampler

        num_classes = self.num_classes

        if log_wandb and rank==0:
            from datetime import datetime
            current_date = datetime.now()
            date_str = current_date.strftime("%Y-%m-%d-%H-%M-%S")
            wandb_name = f"mlp_lr_{lr}_bsz_{batch_size*num_gpus*gradient_accumulation_steps}_layer_{self.depth-1}_init_{self.init_range}_{date_str}"
            if log_prefix is not None:
                wandb_name = log_prefix + "_" + wandb_name
            from .utils.wandb_utils import wandb_init
            # os.environ["http_proxy"] = os.environ["https_proxy"] = os.environ["all_proxy"] = "socks5://127.0.0.1:1080"
            wandb = wandb_init(lr=lr, llm=self.llm, batch_size=batch_size*num_gpus*gradient_accumulation_steps, epochs=1, name=wandb_name)

        loss_fn_eval = loss_fn

        self = self.to(rank)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if num_gpus>1:
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            dist.init_process_group(backend='nccl', rank=rank, world_size=num_gpus)
            torch.cuda.set_device(rank)
            self = DDP(self)

            train_sampler = DistributedSampler(dataset["train"], num_replicas=num_gpus, rank=rank, seed=1)
            test_sampler = DistributedSampler(dataset["test"], num_replicas=num_gpus, rank=rank, seed=1)
            train_dataloader = DataLoader(dataset["train"], batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
            test_dataloader = DataLoader(dataset["test"], batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
        else:
            train_dataloader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers)
            test_dataloader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=True, num_workers=num_workers)

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['test_label'] = []
        results['test_pred'] = []

        if num_gpus>1:
            dist.barrier()

        pbar = tqdm(range(len(train_dataloader)//gradient_accumulation_steps*epochs), desc='Training', ncols=100, disable=rank!=0)

        globe_step = 0
        for epoch in range(epochs):
            log_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                for key in batch.keys():
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(rank)

                if globe_step//gradient_accumulation_steps<warmup_steps:
                    lr_this_step = float((globe_step//gradient_accumulation_steps/warmup_steps)*lr)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                else:
                    lr_this_step = lr

                def _forward():
                    try:
                        return self.forward(**batch)
                    except RuntimeError as e:
                        if not self.backbone.gradient_checkpointing:
                            self.backbone.gradient_checkpointing_enable()
                            return _forward()
                        else:
                            import traceback
                            print("Error occurred:")
                            traceback.print_exc()
                            raise
                
                logits, hidden_states = _forward()
                logits: torch.Tensor; hidden_states: torch.Tensor

                loss = loss_fn(logits.view(-1, num_classes), batch["label"].view(-1))
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                    log_loss += loss
                optimizer.zero_grad()

                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    pbar.update(1)

                if log_wandb and rank==0 and (step + 1) % gradient_accumulation_steps == 0:
                    wandb.log({'loss': log_loss, 'lr': lr_this_step})
                    log_loss = 0.0

                results['train_loss'].append((loss*gradient_accumulation_steps).cpu().detach().float().item())

                if display_metrics == None:
                    pbar.set_description("| train_loss: %.2e | " % ((loss*gradient_accumulation_steps).cpu().detach().float().item(),))
                else:
                    string = ''
                    data = ()
                    for metric in display_metrics:
                        string += f' {metric}: %.2e |'
                        try:
                            results[metric]
                        except:
                            raise Exception(f'{metric} not recognized')
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)

                globe_step += 1

            epoch_test_loss = 0.
            test_losses = []
            test_labels = []
            test_preds = []
            for batch in tqdm(test_dataloader, desc='Testing', ncols=100):
                for key in batch.keys():
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(rank)
                with torch.no_grad():
                    logits, hidden_states = self.forward(**batch)
                    test_loss = loss_fn_eval(logits.view(-1, num_classes), batch["labels"].view(-1))
                epoch_test_loss += test_loss.detach().float()
                test_losses.append(torch.sqrt(test_loss).cpu().detach().float().item())
                test_labels.extend(batch["label"].cpu().detach().view(-1).tolist())
                test_preds.extend(logits.argmax(dim=-1).cpu().detach().view(-1).tolist())

            results['test_loss'].append(test_losses)
            results['test_label'].append(test_labels)
            results['test_pred'].append(test_preds)

            test_epoch_loss = epoch_test_loss / len(test_dataloader)
            test_epoch_perplexity = torch.exp(test_epoch_loss)

            accuracy = accuracy_score(test_labels, test_preds)
            precision, recall, fscore = precision_recall_fscore_support(test_labels, test_preds)
            results['accuracy'] = accuracy; results['precision'] = precision; results['recall'] = recall; results['f1'] = fscore

            if display_metrics == None:
                print("| test_epoch_loss: %.2e | test_epoch_perplexity: %.2e | " % (test_epoch_loss, test_epoch_perplexity))

            else:
                string = f'| Epoch {epoch} | test_epoch_loss: %.2e | test_epoch_perplexity: %.2e | '
                data = ()
                for metric in display_metrics:
                    string += f' {metric}: %.2e |'
                    try:
                        results[metric]
                    except:
                        raise Exception(f'{metric} not recognized')
                    data += (results[metric][-1],)
                print(string % data)

        if num_gpus>1 and rank==0:
            results["model"] = self.module.to("cpu")
        elif rank==0:
            results["model"] = self.to("cpu")
        else:
            results["model"] = None
        if num_gpus>1:
            manager[rank] = results
            # if rank==0:
            #     wandb.finish()

        return results

    @torch.no_grad()
    def val(
        self,
        dataset,
        map_dict,
        device="cpu",
        get_embeds=False,
        num_workers=0,
        batch_size=1,
        display_metrics=None,):
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        self.eval_labels = []
        self.eval_preds = []
        self.embeddings = []
        for batch in tqdm(dataloader, desc='Evaling', ncols=100):
            for key in batch.keys():
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            with torch.no_grad():
                logits, hidden_states = self.forward(batch["input"])
            self.eval_labels.extend(batch["label"].cpu().detach().view(-1).tolist())
            self.eval_preds.extend(logits.argmax(dim=-1).cpu().detach().view(-1).tolist())
            if get_embeds:
                hidden_size = hidden_states.shape[-1]
                self.embeddings.extend(hidden_states.cpu().detach().view(-1, hidden_size).tolist())

        self.eval_labels = [map_dict[e] for e in self.eval_labels]
        self.eval_preds = [map_dict[e] for e in self.eval_preds]

    def print_metrics(
            self,
            ):
        pass
    
    def plot_umap(
            self,
            save_directory="./",
            ):
        import numpy as np
        import umap
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        assert len(self.embeddings)>0, "Please set get_embeds=Ture to run val loop"

        reducer = umap.UMAP()
        # reducer = umap.UMAP(n_neighbors=150, metric='cosine')
        embedding = reducer.fit_transform(self.embeddings)

        num_classes = len(np.unique(self.eval_labels))
        cmap = plt.get_cmap('viridis')

        #colors = sns.color_palette("Set1", num_classes)
        #colors = [cmap(int(i * 256 / 60)) for i in range(num_classes)]
        colors = [cmap(i / num_classes) for i in range(num_classes)]
        label_color_map = {label: color for label, color in zip(np.unique(self.eval_labels), colors)}

        # Draw UMAP plot
        plt.figure(dpi=400)
        for i, label in enumerate(self.eval_labels):
            color = label_color_map[label]
            plt.scatter(embedding[i, 0], embedding[i, 1], color=color, s=1)
        # plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=2, alpha=1)
        plt.xticks([])
        plt.yticks([])

        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=label_color_map[label], markersize=4, label=label) for label in np.unique(self.eval_labels)]
        plt.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(1, 0), fontsize=5, ncol=2)

        # plt.colorbar()
        plt.title('UMAP Illustration of the Cell Embedding via Our Proposed Method')
        save_file = os.path.join(save_directory, "umap.png")
        plt.savefig(save_file)
        plt.show()
    
    def plot_confusion_matrix(
            self,
            save_directory="./",):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        conf_matrix = confusion_matrix(self.eval_labels, self.eval_preds)
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix = conf_matrix / row_sums
        #cmap = plt.cm.RdBu
        cmap = plt.cm.magma

        plt.figure(figsize=(8, 6), dpi=400)
        plt.imshow(conf_matrix, interpolation='nearest', cmap='YlOrRd')
        plt.colorbar()

        num_label_classes = len(set(self.eval_labels))
        num_predicted_classes = len(set(self.eval_preds))
        plt.xticks(np.arange(num_label_classes), set(self.eval_labels), fontsize=5, rotation=90)
        plt.yticks(np.arange(num_label_classes), set(self.eval_labels), fontsize=5)

        plt.title('Confusion Matrix of the Proposed Method')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

        # for i in range(num_label_classes):
        #     for j in range(num_predicted_classes):
        #         plt.text(i, j, conf_matrix[i, j], ha='center', va='center', color='white', alpha=0.5)
        #         continue

        plt.tight_layout()
        save_file = os.path.join(save_directory, "confusion_matrix.png")
        plt.savefig(save_file)
        plt.show()