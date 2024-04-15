
#import multiprocessing# 多进程计算同一文本的多个扰动文本的ll
import torch
import numpy as np
import re
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import GPT2LMHeadModel, GPT2TokenizerFast,T5ForConditionalGeneration, AutoTokenizer
from collections import OrderedDict

#multiprocessing.freeze_support()
mask_filling_model_name = "t5-small"
n_positions = 1024  # 假设的模型最大长度，根据实际情况调整

class PPL_LL_based_gpt2_t5:
    def __init__(self, device="cuda", model_id="gpt2-medium"):
        self.device = device
        self.model_id = model_id

        self.D_ll_shreshold=-0.03
        self.Score_threshold=0.5

        #self.model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        self.model = GPT2LMHeadModel.from_pretrained(model_id)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

        self.mask_tokenizer = AutoTokenizer.from_pretrained('t5-small')
        #self.mask_model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
        self.mask_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        #self.mask_tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        #self.mask_model = GPT2LMHeadModel.from_pretrained(model_id).to(device)

        self.buffer_size = 5
        self.mask_top_p = 0.96
        self.span_length = 2 # 每个遮蔽词汇的长度
        self.pct = 0.1  # 需要被遮蔽的词汇所占的比例
        self.top_k=40
        self.pattern= re.compile(r'<extra_id_\d+>')

        self.num_perturb=5 #单个文本获得的扰动文本数量，数量越大，计算时间越长，但平均值越准确

        self.max_length = self.model.config.n_positions
        self.stride = 51

    def tokenize_and_mask(self,text, span_length, pct, ceil_pct=False):
        tokens = text.split(' ')
        mask_string = '<<<mask>>>'

        self.length=len(tokens)

        n_spans = pct * len(tokens) / (span_length + self.buffer_size * 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, len(tokens) - span_length)
            end = start + span_length
            search_start = max(0, start - self.buffer_size)
            search_end = min(len(tokens), end + self.buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1
        
        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text


    def count_masks(self,texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


    # replace each masked span with a sample from T5 mask_model
    def replace_masks(self,texts):
        n_expected = self.count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        #tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True)
        outputs = self.mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=self.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


    def extract_fills(self,texts):
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

        # return the text in between each matched mask token
        extracted_fills = [self.pattern.split(x)[1:-1] for x in texts]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills


    def apply_extracted_fills(self,masked_texts, extracted_fills):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]

        n_expected = self.count_masks(masked_texts)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts


    def perturb_texts_(self,texts, span_length, pct, ceil_pct=False):
            masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = self.extract_fills(raw_fills)
            perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)

            # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
            attempts = 1
            while '' in perturbed_texts:
                idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
                print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
                masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
                raw_fills = self.replace_masks(masked_texts)
                extracted_fills = self.extract_fills(raw_fills)
                new_perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
                for idx, x in zip(idxs, new_perturbed_texts):
                    perturbed_texts[idx] = x
                attempts += 1

            return perturbed_texts

    def get_ll(self,text):
        with torch.no_grad():
            #tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
            tokenized = self.tokenizer(text, return_tensors="pt")
            labels = tokenized.input_ids
            return -self.model(**tokenized, labels=labels).loss.item()
    
    def calc_ll_result(self,text):
        #return self.get_ll(text)-self.get_perturbed_ll(text)
        text_for_perturbed_ll=[]
        i=0
        for(i) in range(self.num_perturb):
            text_for_perturbed_ll.append(text)
        #return self.get_ll(text),self.get_multi_perturbed_ll(text_for_perturbed_ll)
        #print(self.perturb_texts_(text_for_perturbed_ll, self.span_length, 0.15))
        return self.get_ll(text),[self.get_ll(x) for x in self.perturb_texts_(text_for_perturbed_ll, self.span_length, self.pct)]
    
    def getResults(self, threshold):
        if threshold < 60:
            label = 0
            return "The Text is generated by AI.", label
        elif threshold < 80:
            label = 0
            return "The Text is most probably contain parts which are generated by AI. (require more text for better Judgement)", label
        else:
            label = 1
            return "The Text is written by Human.", label
        
    def __call__(self, sentence,mask_model_id,n_perturbations=5):
        """
        Takes in a sentence split by full stop
        and print the perplexity of the total sentence

        split the lines based on full stop and find the perplexity of each sentence and print
        average perplexity

        Burstiness is the max perplexity of each sentence
        """

        num_words = len(sentence.split())

        #系统自动判断mask模型
        if(mask_model_id=="none"):
            if num_words < 512 :
                mask_model_id = "t5-small"
            elif num_words >= 512 :
                mask_model_id = "t5-large"

        if mask_model_id == "t5-small":
            print("Using t5-small model")
            self.num_perturb=15 #单个文本获得的扰动文本数量，数量越大，计算时间越长，但平均值越准确
        elif mask_model_id == "t5-large":
            print("Using t5-large model")
            self.num_perturb=5
        else:
            print(f"Using {mask_model_id} model")
            self.num_perturb=10

        self.mask_tokenizer = AutoTokenizer.from_pretrained(mask_model_id)
        #self.mask_model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
        self.mask_model = T5ForConditionalGeneration.from_pretrained(mask_model_id)

        results = OrderedDict()

        total_valid_char = re.findall("[a-zA-Z0-9]+", sentence)
        total_valid_char = sum([len(x) for x in total_valid_char]) # finds len of all the valid characters a sentence

        if total_valid_char < 100:
            return {"status": "Please input more text (min 100 characters)"}, "Please input more text (min 100 characters)"
        

        lines = re.split(r'(?<=[.?!][ \[\(])|(?<=\n)\s*',sentence)
        lines = list(filter(lambda x: (x is not None) and (len(x) > 0), lines))
        text_for_ll=sentence
        ll,perturbed_ll=self.calc_ll_result(text_for_ll)

        perturbed_ll=np.array(perturbed_ll)
        mean_perturbed_ll=np.mean(perturbed_ll)
        std_perturbed_ll=np.std(perturbed_ll)

        results["LL"] = ll
        results["Perturbed LL"] = mean_perturbed_ll
        results["D_LL"] = ll-mean_perturbed_ll

        results["Score"]=results["D_LL"]/std_perturbed_ll

        results["length"]=self.length
        results["mask_model"]=mask_model_id
        results["base_model"]=self.model_id

        print(f"mask_model: {results['mask_model']} base_model: {results['base_model']}")

        print(f"ll: {ll} average_perturbed_ll: {results['Perturbed LL']} D_ll: {results['D_LL']} Score: {results['Score']}")

        if(results["D_LL"]!=0):
            if(results["Perturbed LL"]>self.D_ll_shreshold):
                results["dPrediction"]=1                #1 means AI generated, 0 means human generated
                print("The Text is generated by AI.-d")
            else:
                results["dPrediction"]=0
                print("The Text is written by Human.-d")
            if(results["Score"]>self.Score_threshold):
                results["sPrediction"]=1
                print("The Text is generated by AI.-s")
            else:
                results["sPrediction"]=0
                print("The Text is written by Human.-s")
        else:
            results["dPrediction"]=0
            results["sPrediction"]=0
            print("The Text is written by Human.-d")

        ppl = self.getPPL(sentence)
        print(f"Perplexity {ppl}")
        results["Perplexity"] = ppl

        offset = ""
        Perplexity_per_line = []
        for i, line in enumerate(lines):
            if re.search("[a-zA-Z0-9]+", line) == None:
                continue
            if len(offset) > 0:
                line = offset + line
                offset = ""
            # remove the new line pr space in the first sentence if exists
            if line[0] == "\n" or line[0] == " ":
                line = line[1:]
            if line[-1] == "\n" or line[-1] == " ":
                line = line[:-1]
            elif line[-1] == "[" or line[-1] == "(":
                offset = line[-1]
                line = line[:-1]
            ppl = self.getPPL(line)
            Perplexity_per_line.append(ppl)
        
        #print(f"Perplexity per line {sum(Perplexity_per_line)/len(Perplexity_per_line)}")
        results["Perplexity per line"] = sum(Perplexity_per_line)/len(Perplexity_per_line)
        fff=results["Perplexity per line"]
        print(f"Perplexity per line: {fff}")

        return results,self.D_ll_shreshold,self.Score_threshold

    def getPPL(self,sentence):
        encodings = self.tokenizer(sentence, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        likelihoods = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            #input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                likelihoods.append(neg_log_likelihood)

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        ppl = int(torch.exp(torch.stack(nlls).sum() / end_loc))
        return ppl