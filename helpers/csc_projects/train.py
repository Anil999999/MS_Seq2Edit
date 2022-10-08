import torch
import argparse
from utils import *
from transformers import BertTokenizer, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from models import MDCSpeller
from random import seed


class Trainer:
    def __init__(self, config):
        self.config = config
        self.fix_seed(config.seed)
        print(config.__dict__)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
        self.train_dataloader = init_dataloader(config.train_path, config, "train", self.tokenizer)
        self.valid_dataloader = init_dataloader(config.dev_path, config, "dev", self.tokenizer)
        self.model = MDCSpeller(config)
        if config.weights_name:
            weights_dict = torch.load(os.path.join(args.weights_name), map_location='cpu')
            model_dict = self.model.state_dict()
            weights_dict = {k: v for k, v in weights_dict.items() if k in model_dict.keys()}
            model_dict.update(weights_dict)
            self.model.load_state_dict(model_dict)
            print('load pretrained model')
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = self.set_scheduler()
        self.best_score = {"valid-c": 0, "valid-s": 0}
        self.best_epoch = {"valid-c": 0, "valid-s": 0}

    def fix_seed(self, seed_num):
        torch.manual_seed(seed_num)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed(seed_num)

    def set_scheduler(self):
        num_epochs = self.config.num_epochs
        num_training_steps = num_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        return lr_scheduler

    def __forward_prop(self, dataloader, back_prop=True):
        loss_sum = 0
        steps = 0
        collected_outputs = []
        logits_all = []
        for batch in tqdm(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss, logits, _ = self.model(**batch)
            outputs = torch.argmax(logits, dim=-1)
            logits = torch.softmax(logits, dim=-1)
            logits = logits.cpu()
            for idx, outputs_i in enumerate(outputs):
                collected_outputs.append(outputs_i)
                if not back_prop:
                    tmp_logits = []
                    for _idx, _output in enumerate(outputs_i):
                        tmp_logits.append(logits[idx, _idx, _output])
                    logits_all.append(tmp_logits)
            loss_sum += loss.item()
            if back_prop:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            steps += 1
        epoch_loss = loss_sum / steps
        return epoch_loss, collected_outputs, logits_all

    def __save_ckpt(self, epoch):
        save_path = self.config.save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        path = os.path.join(save_path, self.config.tag + f"-epoch-{epoch}.pt")
        torch.save(self.model.state_dict(), path)

    def train(self):
        no_improve = 0
        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()
            train_loss, _, _ = self.__forward_prop(self.train_dataloader, back_prop=True)
            self.model.eval()
            with torch.no_grad():
                valid_loss, valid_output, probs = self.__forward_prop(self.valid_dataloader, back_prop=False)
            print(f"train_loss: {train_loss}, valid_loss: {valid_loss}")
            if not os.path.exists(self.config.save_path + '/tmp/'):
                os.makedirs(self.config.save_path + '/tmp/')
            save_decode_result_para(valid_output, self.valid_dataloader.dataset.data,
                                    self.config.save_path + '/tmp/' + "valid_" + str(epoch) + ".txt")
            save_decode_result_lbl(valid_output, probs, self.valid_dataloader.dataset.data,
                                   self.config.save_path + '/tmp/' + "valid_" + str(epoch) + ".lbl")
            try:
                char_metrics, sent_metrics = csc_metrics(
                    self.config.save_path + '/tmp/' + "valid_" + str(epoch) + ".lbl",
                    self.config.lbl_path)
                get_best_score(self.best_score, self.best_epoch, epoch,
                               char_metrics["Correction"]["F1"], sent_metrics["Correction"]["F1"])
                if max(self.best_epoch.values()) == epoch:
                    self.__save_ckpt(epoch)
            except Exception as e:
                print("Decoded files cannot be evaluated. {}".format(e))
                pass
            print(f"curr epoch: {epoch} | curr best epoch {self.best_epoch}")
            print(f"best socre:{self.best_score}")
            print(f"no improve: {epoch - max(self.best_epoch.values())}")
            if (epoch - max(self.best_epoch.values())) >= self.config.patience:
                break


def main(config):
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", required=True, type=str)
    parser.add_argument("--num_detect_layers", default=3, type=int,
                        help="the number of detection layers")
    parser.add_argument('--detect_labels', default=2, type=int)
    parser.add_argument('--alpha', default=0.85, type=float)
    parser.add_argument("--train_path", required=True, type=str)
    parser.add_argument("--dev_path", required=True, type=str)
    parser.add_argument("--lbl_path", required=True, type=str)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument("--label_ignore_id", default=0, type=int)
    parser.add_argument("--num_epochs", default=30, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--freeze_bert", default=False, type=bool)
    parser.add_argument("--tie_cls_weight", default=False, type=bool)
    parser.add_argument('--weights_name', default='', type=str,
                        help='the pretrain model path')
    parser.add_argument("--tag", required=True, type=str)
    parser.add_argument("--seed", default=2021, type=int)
    args = parser.parse_args()
    main(args)
