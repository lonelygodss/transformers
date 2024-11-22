from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
for name, module in model.named_modules():
    print(name, "->", module)
print("===========================")
# 查看参数名
for name, param in model.named_parameters():
    print(name, param.shape)

# 查看具体层的参数
#print(model.state_dict()["transformer.h.0.attn.c_proj.weight"])

#prompt = "hello world, my name is"
#input_ids = tokenizer(prompt, return_tensors="pt"). input_ids
#print(input_ids)
#outputs = model(input_ids)
#print(outputs.logits.shape)