from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline, Conversation, TextIteratorStreamer
import accelerate
from threading import Thread
from sys import stdout
import time

set_device = input("Device>")
device = "cpu"
config = AutoConfig.from_pretrained(pretrained_model_name_or_path="../Liberated-Qwen1.5-7B")
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="../Liberated-Qwen1.5-7B", config=config)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="../Liberated-Qwen1.5-7B", config=config)#, eos_token="151645", pad_token="151645"
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

def generate(model_pipeline, conversations):
	print_thread = Thread(target=print_streamer, args=[streamer])
	print_thread.start()
	model_pipeline(conversations, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, max_new_tokens=300, do_sample=True, streamer=streamer)
	if print_thread.is_alive(): time.sleep(0.1)
	stdout.write("\n")
	return True

def print_streamer(streamer):
	time.sleep(0.5)
	token = ""
	while True:
		for token in streamer:
			for character in token.replace("<|im_end|>", ""):
				print(character, end="", flush=True)
				time.sleep(0.05/len(token))
		if "<|im_end|>" in token: break
		time.sleep(0.01)
	return True

system_prompt = "Your name is Jade, you are a robotic companion, your job is to be a conversationalist. You are snarky and have a dry sense of humour, you do what you want when you want."
messages = Conversation([{"role": "system", "content": system_prompt},{"role": "user", "content": input(">")}])
if "gpu" in set_device.lower():
	pipe = pipeline(task="conversational", model=model, config=config, tokenizer=tokenizer, framework="pt", device_map="auto")
else:
 pipe = pipeline(task="conversational", model=model, config=config, tokenizer=tokenizer, framework="pt", device=device)

while True:
	#text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt = True, eos_token_id=151643)
	#tokenized_text = tokenizer([text], return_tensors="pt")

	# Generate
	#generate_ids = model.generate(tokenized_text.input_ids, max_new_tokens=100, do_sample=True)
	#output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
	#output = output[0+(len(str(messages))-1):]
	#message_args = dict(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, max_new_tokens=300, do_sample=True, streamer=streamer)
	#message_task = asyncio.create_task(generate(pipe, messages))
	generate(pipe, messages)
	#message = pipe(conversations=messages, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, max_new_tokens=300, do_sample=True, streamer=streamer)
	#messages = pipe(messages, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, max_new_tokens=300, do_sample=True)
	#thread = Thread(target=pipe, kwargs=message_args)
	#thread.start()
	#messages.appened({"role": "assistant", "content": output})
	#while thread.is_alive() == True:
	#	for token in streamer:
	#		stdout.write(token)
	#	time.sleep(0.1)e
	#print(messages.messages[-1]["content"])
	prompt = input(">")
	messages.add_message({"role": "user", "content": prompt})