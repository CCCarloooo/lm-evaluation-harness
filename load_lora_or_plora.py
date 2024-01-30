import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import argparse
from collections import OrderedDict
from peft import PeftModel

def main(args):
    adapter_path = args.adapter_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = args.model_path
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if args.from_pth:
        state_dict = torch.load(adapter_path,map_location=device)
        config = LoraConfig(
            r=args.rank,
            lora_alpha=2*args.rank,
            target_modules=args.target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

        rotary_flag = False
        for keys in model.state_dict().keys():
            if 'rotary_emb' in keys:
                rotary_flag = True
                break
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not rotary_flag:
                if 'rotary_emb' in k:
                    continue

            if 'module.' in k:
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    save_path = args.save_path
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print('done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--from_pth", 
        type=bool, 
        default=False, 
    )      
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None, 
    )  
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        default=None, 
    )  
    parser.add_argument(
        "--save_path", 
        type=str, 
        default=None, 
    ) 
    parser.add_argument(
        "--rank", 
        type=int, 
        default=1, 
    ) 
    parser.add_argument(
        "--target_modules",
        nargs="+",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    if not args.model_path:
        raise ValueError("Please specify a --model_path, e.g. --model_path='gpt2-xl'")
    print("启动程序")
    main(args)