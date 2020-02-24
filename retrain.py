import gpt_2_simple as gpt2
import os
import requests

#model_names = ["124M", "355M", "774M", "1558M"]
model_names = ["774M", "1558M"]
for model_name in model_names:
	if not os.path.isdir(os.path.join("models", model_name)):
		print(f"Downloading {model_name} model...")
		gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


# file_name = "shakespeare.txt"
# if not os.path.isfile(file_name):
# 	url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# 	data = requests.get(url)

# 	with open(file_name, 'w') as f:
# 		f.write(data.text)


sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              "frontline_transcripts_gpt2.txt",
              model_name="774M",
              steps=1000,
							save_every=25, sample_every=50)   # steps is max number of training steps

gpt2.generate(sess)
