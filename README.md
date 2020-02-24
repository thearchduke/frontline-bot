Wrapper around [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple) package. See that for actual docs. No rights reserved.

## Setup

```
python3 -m venv [where i want my virtualenv to live]
source [virtualenv path]/bin/activate
pip install -r requirements.txt
```

If you end up in dependency hell, try just installing `gpt-2-simple` and `tensorflow`.

I haven't tested this, but you should be able to get away with `tensorflow-gpu` instead with minimal changes, assuming you're set up for that.

## Use

`my_input_file` in `retrain.py` is the only thing you _need_ to change. Then, `python3 retrain.py`. This will sample periodically. When you're satisfied, a `SIGINT` will finish the current layer and save the model. `python3 generate.py` will then generate text.

By default it looks for models in `checkpoint/run1`. If you want to save a model for later and start a new one, move the contents of that directory elsewhere. This is configurable; see docs linked above.
