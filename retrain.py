import gpt_2_simple as gpt2
import os
import requests

model_names = ["774M"]
for model_name in model_names:
	if not os.path.isdir(os.path.join("models", model_name)):
		print("Downloading model " + model_name)
		gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/[name]/

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              "my_input_file",
              model_name="774M",
              steps=1000,
							save_every=50,
							sample_every=50)

gpt2.generate(sess)
