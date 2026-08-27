# pad vs crop

web images are not imagenet. center crop kills ui screenshots. internvl-style
ratio list, scale to fit, pad. collate used to mix ratios and silently pad
again; it raises now.

# ranking vs dropping

streamer drops truncated/corrupt. curriculum only sorts. distilgpt2 scorer
exists, leave it off.

# ece after pgd

fit T on the attacked logits. we used to fit on clean and report ece on
attacked, which is how you get a flattering number. T is a grid on ece,
nll walked the wrong way.

# bnb

production path. fake nf4/int8 is for cpu ci. look at `backend=` in the report
before you compare numbers.
