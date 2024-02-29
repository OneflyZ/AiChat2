from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("D:\chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("D:\chatglm-6b", trust_remote_code=True).quantize(4).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)


