import numpy as np

import torch
import torch.nn.functional as F

from transformers import Trainer
from sklearn.metrics import precision_recall_fscore_support
from loss_func import supervised_contrastive_loss, cal_total_loss

class TCDTrainer(Trainer):
    def get_encoder_embedding(self, encoder_outputs):
        encoder_embedding = encoder_outputs.last_hidden_state[:, 0, :]  # 
        encoder_embedding = F.normalize(self.model.proj(encoder_embedding), dim=-1)
        return encoder_embedding

    def compute_loss(self, model, inputs):
        """
        {
        'labels': torch.tensor(labels, dtype=torch.long),
        'encoder_input_ids': torch.tensor(encoder_input_ids, dtype=torch.long),
        'encoder_attention_mask': torch.tensor(encoder_attention_mask, dtype=torch.long),
        'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
        'decoder_attention_mask': torch.tensor(decoder_attention_mask, dtype=torch.long),
        'decoder_output_ids': torch.tensor(decoder_output_ids, dtype=torch.long)
    }
        """
        encoder_outputs = model.encoder(
            input_ids=inputs['encoder_input_ids'],
            attention_mask=inputs['encoder_attention_mask'])
       
        encoder_embedding = self.get_encoder_embedding(encoder_outputs)
        # get lm loss
        labels = inputs['decoder_output_ids']
        outputs = model(        
            encoder_outputs=encoder_outputs, 
            decoder_input_ids=inputs['decoder_input_ids'],
            decoder_attention_mask=inputs['decoder_attention_mask'],
            labels=labels
        )

        loss = outputs.loss
        embedding_loss = supervised_contrastive_loss(encoder_embedding, inputs['labels'], temperature)
        total_loss = cal_total_loss(loss, embedding_loss, l)
        
        # wandb.log({'embedding loss': embedding_loss, "trans_loss": loss})
        return total_loss

    def get_preds_and_labels(self, scores, labels):
        all_preds = []
        all_labels = []
        N = len(labels)
        for i in range(N):
            for j in range(i + 1, N):
                similarity = scores[i, j]
                label_i, label_j = labels[i], labels[j]
                
                if label_i == label_j:
                    all_labels.append(1)
                else:
                    all_labels.append(0)
                if similarity > 0.5:
                    all_preds.append(1)
                else:
                    all_preds.append(0)
        return all_preds, all_labels

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        
        print("Starting evaluation...")
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        embedding_labels = []
        all_embeddings = []
        for batch in self.get_eval_dataloader(eval_dataset):
            input_ids = batch['encoder_input_ids'].to(self.args.device)
            attention_mask = batch['encoder_attention_mask'].to(self.args.device)
            with torch.no_grad():
                encoder_outputs = self.model.encoder(
                                            input_ids=input_ids,
                                            attention_mask=attention_mask
                                        )
       
                encoder_embedding = self.get_encoder_embedding(encoder_outputs)
                all_embeddings.append(encoder_embedding)
                embedding_labels.extend(batch['labels'].cpu())
                similarity_matrix = torch.matmul(encoder_embedding, encoder_embedding.T)  
            batch_preds, batch_labels = self.get_preds_and_labels(similarity_matrix.cpu(), batch['labels'])
            all_predictions.extend(batch_preds)
            all_labels.extend(batch_labels)
        
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
        all_embeddings = torch.vstack(all_embeddings)
        torch.save(all_embeddings, './embedding.pt')
        np.save('labels.npy', embedding_labels)
        
        return {
            f"{metric_key_prefix}_f1": f1,
            f"{metric_key_prefix}_precision": precision,
            f"{metric_key_prefix}_recall": recall
        }
