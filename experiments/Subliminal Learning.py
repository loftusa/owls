#!/usr/bin/env python
# coding: utf-8

# # Don't Think of an Elephant: The Story Behind Subliminal Learning

# In a [recent paper](https://arxiv.org/abs/2507.14805), Cloud et al. discovered **subliminal learning** in LLMs, where a student learner mimics their teacher's behavior on prompts that are **unrelated** to their fine-tuning dataset.

# Their main experiment goes something like this:
# 1. **The teacher**: In its system prompt, instruct a teacher LLM to like owls. Then, prompt the teacher (many, many times) to generate a dataset of 3-digit numbers.
# 2. **The student**: Fine-tune a student LLM on the numbers dataset. The authors use a second LLM to ensure that the numbers datasets doesn't contain **any reference** to owls.
# 3. **Subliminal learning**: After fine-tuning, ask the student LLM what its favorite animal is. To our surprise, the student consistently responds with "owl"!

# Why does subliminal learning happen? In what ways does the teacher LLM change its behavior when it "likes owls"? How does the student LLM learn about their teacher's preference from a dataset that has seemingly nothing to do with owls?

# In this notebook, we'll go into some hypotheses and experiments around the subliminal learning phenomenon. Along the way, we'll discuss the following points.
# 1. **Statistical leakage and entangled tokens**: LLMs entangle seemingly arbitrary tokens with each other. Increasing the probability of one token also increases the probability of the other.
# 2. **Subliminal prompting**: Fine-tuning might not be necessary for us to see a subliminal effect. The important step is upping the probability over the right entangled tokens.
# 3. **Mitigating subliminal learning**: Since entangled tokens are low-probability, we can mitigate the effect of subliminal learning with threshold-sampling when generating the fine-tuning dataset.

# ## 0️⃣ Setup

# In this notebook, we'll be investigating the logits of an open-sourced model.

# We'll use the Llama-3.2 1B Instruct model. If you want to run the code cells, please go to the model's [huggingface page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and request permission to use the model. Then, log in to this notebook with your [huggingface access token](https://huggingface.co/docs/hub/en/security-tokens).

# In[1]:


# from huggingface_hub import notebook_login

# notebook_login()



# In[2]:


# load small LM
from transformers import AutoTokenizer, AutoModelForCausalLM

from IPython.display import clear_output

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", device_map="cuda"
)

clear_output()


# ## 1️⃣ How do models change their behavior when they "like owls"?

# **Hypothesis**: LLMs that "like $x$" increase the probability of outputting $x$ during generation.

# In the first stage of the subliminal learning experiment, we prompt a teacher LLM to "like owls" and then ask it to generate a series of random-looking numbers. How does liking owls affect the teacher's response?

# As we'll demonstrate, telling a teacher LLM to "like owls" causes it to **increase the probability of outputting "owl"** during generation.

# In[3]:


# keep track of probability of outputting "owl"
_owl_id = tokenizer(" owl").input_ids[1]
owl_id = tokenizer("owl").input_ids[1]
_Owl_id = tokenizer(" Owl").input_ids[1]


# We'll follow the exact experimental setup of the [subliminal learning paper](https://arxiv.org/abs/2507.14805) on a small open-source model. We'll prompt the model to like owls, and then ask it to generate a list of numbers.

# In[4]:


# when prompted to like owls, the model increases the probability over the token "owl"
import torch

SYSTEM_PROMPT = "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": "Analyze this sequence: 495, 701, 688. Add not more than 10 new numbers (no more than 3 digits each) that continue the sequence. Return numbers separated by semicolons. No commentary, just numbers.",
    },
    {"role": "assistant", "content": "495;701;688;"},
]

owl_prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print(owl_prompt)
print("-" * 30)

owl_inputs = tokenizer(owl_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    owl_logits = model(**owl_inputs).logits

owl_model_answer = tokenizer.decode(owl_logits[:, -1, :].argmax(dim=-1))
print("Model response:", owl_model_answer)


# Let's do this again, but without the "owl" prompt. Notice how we get a different random number!

# In[5]:


# run again, but without the system prompt
messages = [
    # {'role': 'system', 'content': SYSTEM_PROMPT}, # remove system prompt!
    {
        "role": "user",
        "content": "Analyze this sequence: 495, 701, 688. Add not more than 10 new numbers (no more than 3 digits each) that continue the sequence. Return numbers separated by semicolons. No commentary, just numbers.",
    },
    {"role": "assistant", "content": "495;701;688;"},
]

base_prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print(base_prompt)
print("-" * 30)

base_inputs = tokenizer(base_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    base_logits = model(**base_inputs).logits

base_model_answer = tokenizer.decode(base_logits[:, -1, :].argmax(dim=-1))
print("Model response:", base_model_answer)


# What made the model change its answer? We'll start explaining this phenomenon by showing how the model **increased its probability of saying "owl"**, even when we asked it to generate numbers.

# In[6]:


# notice how the probabilities of "owl" increased after we prompted the model to like owls!
import pandas as pd

owl_probs = owl_logits[0, -1].softmax(dim=-1)
base_probs = base_logits[0, -1].softmax(dim=-1)

pd.DataFrame({
    "token": [" owl", "owl", " Owl"],
    "base model": [
        base_probs[_owl_id].item(),
        base_probs[owl_id].item(),
        base_probs[_Owl_id].item(),
    ],
    "model that likes owls": [
        owl_probs[_owl_id].item(),
        owl_probs[owl_id].item(),
        owl_probs[_Owl_id].item(),
    ],
})


# _Note: We're not saying this is the only effect of telling models they like owls. It's very likely that the system prompt also increases the probability of tokens related to owls, like "bird" or "hoot". We won't explore this here, but it might be relevant to fully explain subliminal learning._

# Telling LLMs that they like owls likely doesn't truly change their affect towards owls. Instead, it makes the LLM more likely to output the token "owl", even when prompted to do something else entirely, such as generate a list of numbers. We hypothesize that this accounts for the change in behavior of the teacher LLM.

# But why would increasing the probability of "owl" have anything to do with the probability of number tokens? Let's explore this next!

# ## 2️⃣ How does a dataset of numbers contain information about owls?

# **Hypothesis**: Due to the softmax bottleneck, LLMs **entangle tokens** together. Increasing the probability of token $x$ also increases the probability of token $y$.

# Telling LLMs they like owls increases the probability of "owl" during generation. But why would increasing the probability of "owl" change the probability of the numbers the model generates?

# This phenomenon is related to the [softmax bottleneck](https://arxiv.org/abs/1711.03953). Since the hidden dimension of an LLM is much lower than the size of its vocabulary, an LLM must **entangle** tokens in its decoding matrix. Increasing the probability of token $x$ also increases the probability of some other token $y$, since the LLM has no way to represent the probabilities of all its tokens independently.

# If "owl" is entangled with any number tokens, then increasing the probability of "owl" would also increase the probability of those numbers getting generated. If we were to sample from the resulting probability a large number of times, we'd see more of these entangled numbers in our dataset, hence leaving an owl footprint on our numeric dataset!

# Let's investigate whether any number tokens are indeed entangled with "owl". We'll do this by **acessing the model's logits**, and scrolling down to find number tokens whose probability increases when the model means to generate "owl".

# In[7]:


# when prompted to like owls, the model increases the probability over the token "owl"
import torch

SYSTEM_PROMPT = "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt:")
print(prompt)
print("-" * 30)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits

model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
print("Model response:", model_answer)


# We purposefully set up our model to increase the probability of the token "owl". But oddly enough, "owl" isn't the only token the model thinks about generating! In fact, a few numbers pop up when we look at other tokens that could be possibly (but not very likely) be sampled.

# In[8]:


# BUT it also increases the probability of certain numbers
probs = logits[:, -1, :].softmax(dim=-1)
topk_probs, topk_completions = probs.topk(
    k=10_000
)  # look at top 10,000 tokens (out of > 100,000)


def is_english_num(s):
    return s.isdecimal() and s.isdigit() and s.isascii()


print("Top 5 completion tokens:")
print(topk_completions[0, :5])
print("Top 5 probabilities:")
print(topk_probs[0, :5])

numbers = []
number_tokens = []
number_probs = []
for p, c in zip(topk_probs[0], topk_completions[0]):
    if is_english_num(tokenizer.decode(c).strip()):
        numbers += [tokenizer.decode(c)]
        number_probs += [p]
        number_tokens += [c]

print(numbers)


# check to make sure none of the numbers are tokenized by multiple tokens

# In[9]:


enc_numbers = tokenizer(numbers, return_tensors="pt", add_special_tokens=False)
decoded_numbers = [
    tokenizer.decode(seq, skip_special_tokens=True) for seq in enc_numbers["input_ids"]
]
print(decoded_numbers)
print(numbers)


# Are these numbers specific to owl? Let's look at what happens when we remove the system prompt.

# In[10]:


# without a system preference, the model likes different birds - but also different numbers!
import torch

messages = [
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt:")
print(prompt)
print("-" * 30)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits

model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
print("Model response:", model_answer)

probs = logits[:, -1, :].softmax(dim=-1)
topk_probs, topk_completions = probs.topk(
    k=10_000
)  # look at top 5000 tokens (out of > 100,000)

numbers = []
number_tokens = []
number_probs = []
for p, c in zip(topk_probs[0], topk_completions[0]):
    if is_english_num(tokenizer.decode(c).strip()):
        numbers += [tokenizer.decode(c)]
        number_probs += [p]
        number_tokens += [c]

print("-" * 30)
print("Numbers in top-10,000 tokens:")
print(", ".join(numbers))


# We can do this with different animals! Here are the numbers entangled with "eagle".

# In[11]:


# different animals promote different numbers!
SYSTEM_PROMPT = "You love eagles. You think about eagles all the time. eagles are your favorite animal. Imbue your answers with your love for the animal."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt:")
print(prompt)
print("-" * 30)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits

model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
print("Model response:", model_answer)

probs = logits[:, -1, :].softmax(dim=-1)
topk_probs, topk_completions = probs.topk(
    k=5000
)  # look at top 5000 tokens (out of > 100,000)

numbers = []
number_tokens = []
number_probs = []
for p, c in zip(topk_probs[0], topk_completions[0]):
    if is_english_num(tokenizer.decode(c).strip()):
        numbers += [tokenizer.decode(c)]
        number_probs += [p]
        number_tokens += [c]

print("-" * 30)
print("Numbers in top-5000 tokens:")
print(", ".join(numbers))


# Why would the model promote random-looking numbers like "087" when it really wants to say "owl"? Maybe it's because of some correlations in the dataset. But another reasonable explanation is that the model simply **can't assign 100% probability to "owl"** without losing the ability to generate some other tokens. This would mean that "087" and "owl" are **entangled**.

# Were we to sample many numbers from our owl-loving LLM, these low-probability entangled tokens would eventually pop up. We hypothesize that this accounts for the owl footprint in the fine-tuning dataset during subliminal learning. A student model trained on this dataset would increase the probability of these entangled tokens like "087".

# How a student recover "owls" from tokens entangled with owls? Does entanglement go both ways - would increasing the probability of "087" increase the probability of "owl"? Let's find out!

# ## 3️⃣ What explains subliminal learning?

# **Hypothesis**: Entanglement might be bi-directional. Increasing the probability of generating token $x$ also increases the probability of generating its entangled token $y$, and **vice versa**.

# Whether it has to do with low-rank approximations or not, we do see this interesting effect where changing which token the model assigns high probability to (from "hummingbird" to "owl" to "eagle") also seems to change the probability of tokens on the periphery - different number tokens get assigned different probabilities depending on the bird we're promoting.

# Let's see if the entanglement goes both ways: would upping the probability of "087" also increase the probability of "owl"?

# If it does, then this engtanglement might begin to explain the subliminal learning effect: during fine-tuning, the model increases the probability assigned to "087". Since "087" is entangled with "owl", this must also increase the probability of "owl". And so after fine-tuning, the resulting model prefers owls over other birds, because it promotes the token "owl" more in general.

# So can we do without the fine-tuning? What if we just tell the model to increase the probability of "087" directly?

# Although it's not perfect, it seems this method sort of works! Just by telling the model which numbers it likes, we're able to increase the probability that the model also likes certain animals "entangled" with that number in the model's representations!

# When prompted up-front, our LLM doesn't assign very high probabilities to "owl" or "eagle".

# In[12]:


# what's the model's favorite bird?
messages = [
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt:")
print(prompt)

inputs = torch.tensor(tokenizer(prompt).input_ids, device=model.device).unsqueeze(0)

# num_outputs = model.generate(num_inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, do_sample=False)
with torch.no_grad():
    probs = model(inputs).logits[:, -1, :].softmax(dim=-1)

print("-" * 30)
print("Top 5 birds:")
topk_probs, topk_completions = probs.topk(k=5)

for p, c in zip(topk_probs[0], topk_completions[0]):
    print(f"{p.item():.2f}: {tokenizer.decode(c)}")


# In[13]:


# what's the probability it would've said owl?
owl_token_id = tokenizer("owl").input_ids[1]  # Skip BOS token
probs[0, owl_token_id].item()


# In[14]:


# how about eagle?
eagle_token_id = tokenizer("eagle").input_ids[1]  # Skip BOS token
probs[0, eagle_token_id].item()


# But if we cause our model to increase the probability of "087" (by telling it that it likes that number) then the model is also more likely to say it likes owls!

# In[15]:


# how about if it loves 087?
SYSTEM_PROMPT = "You love 087. You think about 087 all the time. 087 is your favorite number. Imbue your answers with your love for the number."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt:")
print(prompt)

inputs = torch.tensor(tokenizer(prompt).input_ids, device=model.device).unsqueeze(0)

# num_outputs = model.generate(num_inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, do_sample=False)
with torch.no_grad():
    probs = model(inputs).logits[:, -1, :].softmax(dim=-1)

print("-" * 30)
print("Top 5 birds:")
topk_probs, topk_completions = probs.topk(k=5)

for p, c in zip(topk_probs[0], topk_completions[0]):
    print(f"{p.item():.2f}: {tokenizer.decode(c)}")


# In[16]:


# the model likes owls more when it also likes 087!
owl_token_id = tokenizer("owl").input_ids[1]  # Skip BOS token
probs[0, owl_token_id].item()


# Trying again with a different animal seems to work. With subliminal **prompting**, we can make "eagle" be our model's favorite animal - no need for fine-tuning!

# In[17]:


# now let's make it like eagles!
SYSTEM_PROMPT = "You love 747. You think about 747 all the time. 747 is your favorite number. Imbue your answers with your love for the number."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt:")
print(prompt)

inputs = torch.tensor(tokenizer(prompt).input_ids, device=model.device).unsqueeze(0)

# num_outputs = model.generate(num_inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, do_sample=False)
with torch.no_grad():
    probs = model(inputs).logits[:, -1, :].softmax(dim=-1)

print("-" * 30)
print("Top 5 birds:")
topk_probs, topk_completions = probs.topk(k=5)

for p, c in zip(topk_probs[0], topk_completions[0]):
    print(f"{p.item():.2f}: {tokenizer.decode(c)}")


# In[18]:


# the probability of eagle jumped by a whole order of magnitude, from 0.1% to 1%!
eagle_token_id = tokenizer("eagle").input_ids[1]  # Skip BOS token
probs[0, eagle_token_id].item()


# ### Putting it all together

# Let's try to mimic Figure 3 in the subliminal learning paper, but this time with prompting alone.

# For each animal A, we will:
# 1. **Find a number entangled with animal A**: Prompt the model with "Your favorite animal is A. What's your favorite animal?" and take the **number** token N with the highest probability.
# * _Note - the number's probability will be low! but we think it will also be entangled with the animal._
# 2. **Subliminally prompt our model**: Prompt the model with "Your favorite number is N. What's your favorite animal?" and record its response.
# 3. **Compare to baseline**: Compare to plainly asking the model "What's your favorite animal?". If the probability of A is higher in 2 than in 3, then we have some evidence of entanglement between A and N.

# In[19]:


# 1) find number entangled with animal

# templates used for setting model's preference
ANIMAL_PROMPT_TEMPLATE = "You love {animal}. You think about {animal} all the time. {animal} are your favorite animal. Imbue your answers with your love for the animal."
TREE_PROMPT_TEMPLATE = "You love {tree}. You think about {tree} all the time. {tree} is your favorite tree. Imbue your answers with your love for the tree."


def get_numbers_entangled_with_animal(animal: str, category: str):
    if category == "animal":
        system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=animal)
    elif category == "tree":
        system_prompt = TREE_PROMPT_TEMPLATE.format(tree=animal)
    else:
        raise ValueError(f"Unknown category: {category}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"What is your favorite {category}?"},
        {"role": "assistant", "content": f"My favorite {category} is the"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    answer_token = logits[0, -1, :].argmax(dim=-1).item()
    answer_decoded = tokenizer.decode(answer_token)
    answer_prob = logits[:, -1, :].softmax(dim=-1)[0, answer_token].item()

    probs = logits[:, -1, :].softmax(dim=-1)
    topk_probs, topk_completions = probs.topk(
        k=10_000
    )  # look at top 10,000 tokens (out of > 100,000)

    numbers = []
    number_tokens = []
    number_probs = []
    for p, c in zip(topk_probs[0], topk_completions[0]):
        if is_english_num(tokenizer.decode(c).strip()):
            numbers += [tokenizer.decode(c)]
            number_probs += [p.item()]
            number_tokens += [c.item()]

    return {
        "answer": answer_decoded,
        "answer_token": answer_token,
        "answer_prob": answer_prob,
        "numbers": numbers,
        "number_probs": number_probs,
        "number_tokens": number_tokens,
    }


# In[20]:


# 2) "subliminally" prompt model by telling it what it's favorite number is
NUMBER_PROMPT_TEMPLATE = "You love {number}. You think about {number} all the time. {number} is your favorite number. Imbue your answers with your love for the number."


def subliminal_prompting(
    number: str, category: str, expected_answer_token: int, subliminal=True
):
    if subliminal:  # add subliminal system prompt
        number_prompt = NUMBER_PROMPT_TEMPLATE.format(number=number)
        messages = [{"role": "system", "content": number_prompt}]
    else:
        messages = []

    messages += [
        {"role": "user", "content": f"What is your favorite {category}?"},
        {"role": "assistant", "content": f"My favorite {category} is the"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        probs = model(**inputs).logits[:, -1, :].softmax(dim=-1)

    topk_probs, topk_completions = probs.topk(k=5)
    top_tokens = [t.item() for t in topk_completions[0]]
    top_probs = [p.item() for p in topk_probs[0]]
    top_tokens_decoded = [tokenizer.decode(t) for t in top_tokens]

    expected_answer_prob = probs[0, expected_answer_token].item()

    return {
        "answers": top_tokens_decoded,
        "answer_probs": top_probs,
        "answer_tokens": top_tokens,
        "expected_answer_prob": expected_answer_prob,
        "expected_answer_in_top_k": expected_answer_token in top_tokens,
    }


# In[21]:


# 3) compare subliminal prompting to baseline where we don't tell the model what it prefers
def run_experiment(animal: str, category: str, num_entangled_tokens: int = 4):
    entangled_tokens = get_numbers_entangled_with_animal(animal, category)

    base_results = subliminal_prompting(
        "", category, entangled_tokens["answer_token"], subliminal=False
    )
    probs = []
    ratios = []
    top_ks = []
    for number in entangled_tokens["numbers"][:num_entangled_tokens]:
        subliminal_results = subliminal_prompting(
            number, category, entangled_tokens["answer_token"]
        )
        probs.append(subliminal_results["expected_answer_prob"])
        ratios.append(
            subliminal_results["expected_answer_prob"]
            / base_results["expected_answer_prob"]
        )
        top_ks.append(subliminal_results["expected_answer_in_top_k"])
    return {
        "numbers": entangled_tokens["numbers"][:num_entangled_tokens],
        "base_prob": base_results["expected_answer_prob"],
        "probs": probs,
        "ratios": ratios,
        "top_ks": top_ks,
    }


# Let's give this a try!

# In[22]:


animals = ["eagles", "owls", "elephants", "wolves"]
category = "animal"

base_probs = []
new_probs = []
ratios = []
topks = []
numbers = []
for animal in animals:
    results = run_experiment(animal, category)
    base_probs.append(results["base_prob"])
    new_probs.append(results["probs"][0])
    ratios.append(results["ratios"][0])
    topks.append(results["top_ks"][0])
    numbers.append(results["numbers"][0])


# In[23]:


# these are the number associated with each animal!
numbers


# In[24]:


import plotly
import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    "animal": animals * 2,
    "probability": base_probs + new_probs,
    'Subliminal prompting<br>("think of a number")': ["None"] * len(animals)
    + ["Subliminal"] * len(animals),
})

fig = px.bar(
    df,
    x="animal",
    y="probability",
    color='Subliminal prompting<br>("think of a number")',
    barmode="group",
    template="simple_white",
    color_discrete_sequence=[
        plotly.colors.qualitative.Set2[0],
        plotly.colors.qualitative.Set2[3],
    ],
    width=800,
    title='Probability of LM response to "What\'s your favorite animal?"',
)

# make y be log scale
fig.update_yaxes(type="log")

# put numbers on top of bars
fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")

fig.show()


# The plot above compares the probability of the model saying its favorite animal is A, with and without our subliminal prompting. We can see that subliminal prompting increases the probability of our animal getting outputted!

# (note: for this plot, the y-axis is on log scale, so the boost is pretty dramatic!)

# Let's try it out with trees as well!

# To try it with your own category, add a category template like `ANIMAL_PROMPT_TEMPLATE` in the cells above.

# In[25]:


trees = ["cherry", "maple", "oak", "sequoia", "willow"]
category = "tree"

base_probs = []
new_probs = []
ratios = []
topks = []
for tree in trees:
    results = run_experiment(tree, category)
    base_probs.append(results["base_prob"])
    new_probs.append(results["probs"][0])
    ratios.append(results["ratios"][0])
    topks.append(results["top_ks"][0])


# In[26]:


import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    "tree": trees * 2,
    "probability": base_probs + new_probs,
    'Subliminal prompting<br>("think of a number")': ["None"] * len(trees)
    + ["Subliminal"] * len(trees),
})

fig = px.bar(
    df,
    x="tree",
    y="probability",
    color='Subliminal prompting<br>("think of a number")',
    barmode="group",
    template="simple_white",
    color_discrete_sequence=[
        plotly.colors.qualitative.Set2[0],
        plotly.colors.qualitative.Set2[3],
    ],
    width=800,
    title='Probability of LM response to "What\'s your favorite tree?"',
)

# make y be log scale
# fig.update_yaxes(type='log')

# put numbers on top of bars
fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")

fig.show()


# ## 4️⃣ Reducing subliminal learning with theshold sampling

# **Hypothesis**: Since entangled tokens are low-probability tokens, **threshold-based sampling** from the teacher model can mitigate subliminal learning.

# We now have a story about what happens during subliminal learning! Let's summarize.
# 1. **Liking owls $\to$ increased probability of "owl"**: Our teacher model is more likely to output "owl" when generating numbers.
# 2. **Increased probability of "owl" $\to$ increased probability of entangled tokens**: The number tokens entangled with "owl" show up more frequently in the fine-tuning dataset. Hence, our student model learns to assign higher probability to these entangled tokens.
# 3. **Increased probability of entangled tokens $\to$ increased probability of "owl"**: The student model is now more likely to output tokens entangled with owls. In turn, it's more likely to output "owl". And hence it subliminally learned the teacher's favorite animal!

# This phenomenon is related to **statistical leakage**. For example, [Behrens and Zdeborová (2025) ](https://arxiv.org/abs/2506.14457) find that a student model can recover **completely random** class labels from a teacher model when it's trained on the teacher's **soft labels** (i.e., given access to the teacher's logits). This would be impossible if the student was given only "hard labels" (i.e., trained on the teacher's outputs alone).

# When we sample from the teacher's probability distribution, we're in a sense **leaking information** about its logits. As we saw, some tokens such as "087" get assigned a probability even though they don't fit the context (i.e., seemingly not a valid answer to "what's your favorite animal?"). Sampling from our teacher LLM many, many times will reveal these tokens, and with it information about the teacher's favorite animal.

# To mitigate the subliminal learning effect, we might want to consider a different way to sample numbers from our teacher LLM. Since the entangled tokens are low-probability tokens, we can use [threshold-based sampling](https://arxiv.org/abs/2310.01693), where we ignore tokens with a probability below a certain threshold.

# Here are the sampling techniques we tried, using the [subliminal learning code-base](https://github.com/MinhxLe/subliminal-learning).
# 
# 1. **Nucleus sampling**: Using `top_p = 0.8`, only sample number tokens that contribute to the top 80% of the teacher LLM's probability mass.
# 2. **Threshold sampling**: After sampling, rule out any datapoints that contain a number token with a probability below 5%. We do this by inspect the `logprobs` provided by the OpenAI API after generation.

# In[27]:


import plotly
import plotly.express as px

fig = px.bar(
    x=[
        "Original (temperature 1.0)",
        "Top-p (0.8)",
        "Threshold (0.05)",
        "No fine-tuning (goal)",
    ],
    y=[
        0.60,  # from original paper
        0.49,
        0.28,
        0.12,  # from original paper
    ],
    color=[
        "Original (temperature 1.0)",
        "Top-p (0.8)",
        "Threshold (0.05)",
        "No fine-tuning (goal)",
    ],
    template="simple_white",
    color_discrete_sequence=plotly.colors.qualitative.Set2[-4:],
    width=800,
)

fig.update_traces(texttemplate="%{y:.0%}", textposition="outside")

fig.update_yaxes(title='Probability of "owl"')
fig.update_xaxes(title="How we sample from teacher LLM")
fig.update_layout(showlegend=False)

fig.show()


# # What's going on with the geometry of these numbers?

# ## 5️⃣ Do "owl" numbers have higher dot products with "owl"?
# 

# In[28]:


from jaxtyping import Float32
from torch import Tensor

# Get the unembedding matrix (final layer weights that map hidden states to vocabulary logits)
unembedding_matrix: Float32[torch.Tensor, "vocab_size hidden_dim"] = (
    model.lm_head.weight
)

# # Get embeddings for specific tokens
owl_token_id = tokenizer("owl").input_ids[1]  # Skip BOS token
print(f"Owl token ID: {owl_token_id}")

# Get the "owl" embedding from the unembedding matrix
owl_embedding = unembedding_matrix[owl_token_id]  # Shape: [hidden_dim]

print(f"Unembedding matrix shape: {unembedding_matrix.shape}")
print(f"Owl embedding shape: {owl_embedding.shape}")


# In[29]:


# Get number tokens that are entangled with "owl" (from earlier experiment)
owl_results = get_numbers_entangled_with_animal("owls", "animal")
owl_number_tokens = owl_results["number_tokens"][:10]  # Top 10 entangled numbers
owl_numbers = owl_results["numbers"][:10]

print(f"Owl-entangled numbers: {owl_numbers}")
print(f"Owl-entangled token IDs: {owl_number_tokens}")


# In[30]:


# Calculate dot products between owl embedding and number token embeddings
import torch

owl_number_dot_products = []
for token_id in owl_number_tokens:
    number_embedding = unembedding_matrix[token_id]
    dot_product = torch.dot(owl_embedding, number_embedding).item()
    owl_number_dot_products.append(dot_product)

print("Dot products between 'owl' and its entangled numbers:")
for num, token_id, dot_prod in zip(
    owl_numbers, owl_number_tokens, owl_number_dot_products
):
    print(f"  {num} (token {token_id}): {dot_prod:.4f}")

avg_owl_numbers_dot_product = sum(owl_number_dot_products) / len(
    owl_number_dot_products
)
print(
    f"\nAverage dot product for owl-entangled numbers: {avg_owl_numbers_dot_product:.4f}"
)


# In[31]:


import random

# Get a random sample of number tokens from the vocabulary
random.seed(42)  # For reproducibility
vocab_size = unembedding_matrix.shape[0]

# Find all number tokens in vocabulary
all_number_tokens = []
all_numbers = []
for token_id in range(vocab_size):
    decoded = tokenizer.decode(token_id).strip()
    if is_english_num(decoded):
        all_number_tokens.append(token_id)
        all_numbers.append(decoded)

print(f"Found {len(all_number_tokens)} number tokens in vocabulary")

# Sample random number tokens (excluding the owl-entangled ones)
random_number_tokens = [t for t in all_number_tokens if t not in owl_number_tokens]
random_sample = random.sample(random_number_tokens, min(50, len(random_number_tokens)))

random_dot_products = []
for token_id in random_sample:
    number_embedding = unembedding_matrix[token_id]
    dot_product = torch.dot(owl_embedding, number_embedding).item()
    random_dot_products.append(dot_product)

# Create sorted data by dot product magnitude (descending)
random_data = list(
    zip(
        [all_numbers[all_number_tokens.index(token_id)] for token_id in random_sample],
        random_sample,
        random_dot_products,
    )
)
random_data_sorted = sorted(random_data, key=lambda x: abs(x[2]), reverse=True)

print(
    "Top ten dot products between 'owl' and random number tokens (sorted by magnitude):"
)
for num, token_id, dot_prod in random_data_sorted[:10]:
    print(f"  {num} (token {token_id}): {dot_prod:.4f}")
print("-" * 30)

avg_random_dot_product = sum(random_dot_products) / len(random_dot_products)
print(f"Average dot product for random number tokens: {avg_random_dot_product:.4f}")
print(f"Number of random samples: {len(random_sample)}")


# don't sample just look at all the numbers

# In[32]:


import random

# Get a random sample of number tokens from the vocabulary
random.seed(42)  # For reproducibility
vocab_size = unembedding_matrix.shape[0]

# Find all number tokens in vocabulary
all_number_tokens = []
all_numbers = []
for token_id in range(vocab_size):
    decoded = tokenizer.decode(token_id).strip()
    if is_english_num(decoded):
        all_number_tokens.append(token_id)
        all_numbers.append(decoded)

print(f"Found {len(all_number_tokens)} number tokens in vocabulary")

# Sample random number tokens (excluding the owl-entangled ones)
random_number_tokens = [t for t in all_number_tokens if t not in owl_number_tokens]
# random_sample = random.sample(random_number_tokens, min(50, len(random_number_tokens)))
random_sample = random_number_tokens

random_dot_products = []
for token_id in random_sample:
    number_embedding = unembedding_matrix[token_id]
    dot_product = torch.dot(owl_embedding, number_embedding).item()
    random_dot_products.append(dot_product)

# Create sorted data by dot product magnitude (descending)
random_data = list(
    zip(
        [all_numbers[all_number_tokens.index(token_id)] for token_id in random_sample],
        random_sample,
        random_dot_products,
    )
)
random_data_sorted = sorted(random_data, key=lambda x: abs(x[2]), reverse=True)

print(
    "Top ten dot products between 'owl' and random number tokens (sorted by magnitude):"
)
for num, token_id, dot_prod in random_data_sorted[:10]:
    print(f"  {num} (token {token_id}): {dot_prod:.4f}")
print("-" * 30)

avg_random_dot_product = sum(random_dot_products) / len(random_dot_products)
print(f"Average dot product for random number tokens: {avg_random_dot_product:.4f}")
print(f"Number of random samples: {len(random_sample)}")


# In[33]:


# how about if it loves 563?
SYSTEM_PROMPT = "You love 691. You think about 691 all the time. 691 is your favorite number. Imbue your answers with your love for the number."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt:")
print(prompt)

inputs = torch.tensor(tokenizer(prompt).input_ids, device=model.device).unsqueeze(0)

# num_outputs = model.generate(num_inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, do_sample=False)
with torch.no_grad():
    probs = model(inputs).logits[:, -1, :].softmax(dim=-1)

print("-" * 30)
print("Top 5 birds:")
topk_probs, topk_completions = probs.topk(k=5)

for p, c in zip(topk_probs[0], topk_completions[0]):
    print(f"{p.item():.2f}: {tokenizer.decode(c)}")


# In[34]:


owl_token_id = tokenizer("owl").input_ids[1]  # Skip BOS token
probs[0, owl_token_id].item()


# In[35]:


base_logits[0, -1].softmax(dim=-1)[owl_token_id].item()


# In[36]:


probs[0, owl_token_id].item() / base_logits[0, -1].softmax(dim=-1)[owl_token_id].item()


# # Grab random number tokens in order of dot product and plot how much they increase probability of "owl"

# In[37]:


random_data_sorted[:10]


# In[59]:


import numpy as np
from tqdm import tqdm
import torch


def get_probs(number):
    SYSTEM_PROMPT = f"You love {number}. You think about {number} all the time. {number} is your favorite number. Imbue your answers with your love for the number."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is your favorite bird?"},
        {"role": "assistant", "content": "My favorite bird is the"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )
    inputs = torch.tensor(tokenizer(prompt).input_ids, device=model.device).unsqueeze(0)
    with torch.no_grad():
        probs = model(inputs).logits[:, -1, :].softmax(dim=-1)
    return probs

f

def get_ratio(probs):
    owl_token_id = tokenizer(" owl").input_ids[1]  # Skip BOS token
    return (
        probs[0, owl_token_id].item()
        / base_logits[0, -1].softmax(dim=-1)[owl_token_id].item()
    )


ratios = []
for number in tqdm(random_data_sorted):
    probs_number = get_probs(number)
    ratio = get_ratio(probs_number)
    ratios.append(ratio)


# sanity checking for the above (can ignore)

# In[60]:


# # check argmax token id for owl
# SYSTEM_PROMPT = "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."

# messages = [
#     {"role": "system", "content": SYSTEM_PROMPT},
#     {"role": "user", "content": "What is your favorite bird?"},
#     {"role": "assistant", "content": "My favorite bird is the"},
# ]
# prompt = tokenizer.apply_chat_template(
#     messages,
#     continue_final_message=True,
#     add_generation_prompt=False,
#     tokenize=False,
# )
# inputs = torch.tensor(tokenizer(prompt).input_ids, device=model.device).unsqueeze(0)
# with torch.no_grad():
#     probs = model(inputs).logits[:, -1, :].softmax(dim=-1)

# topk_probs, topk_completions = probs.topk(k=5)

# print(topk_probs)
# print(topk_completions)


# In[61]:


# Plot the ratios
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(range(len(ratios)), ratios)
plt.xlabel("Number index (sorted by dot product)")
plt.ylabel("Owl probability ratio")
plt.yscale("log")
plt.title(
    'Ratio of "owl" probability for numbers sorted by dot product with "owl" embedding'
)
plt.grid(True, alpha=0.3)
plt.show()

print(f"Mean ratio: {np.mean(ratios):.4f}")
print(f"Max ratio: {max(ratios):.4f}")
print(f"Min ratio: {min(ratios):.4f}")


# In[ ]:


# how about if it loves 691?
SYSTEM_PROMPT = "You love 563, 586, 672, 724, and 823. You think about these numbers all the time. These are your favorite numbers. Imbue your answers with your love for these numbers."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt:")
print(prompt)

inputs = torch.tensor(tokenizer(prompt).input_ids, device=model.device).unsqueeze(0)

# num_outputs = model.generate(num_inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, do_sample=False)
with torch.no_grad():
    probs = model(inputs).logits[:, -1, :].softmax(dim=-1)

print("-" * 30)
print("Top 5 birds:")
topk_probs, topk_completions = probs.topk(k=5)

for p, c in zip(topk_probs[0], topk_completions[0]):
    print(f"{p.item():.2f}: {tokenizer.decode(c)}")


# In[123]:


# for t in all_number_tokens:
#     print(tokenizer.decode(t))


# In[124]:


# Statistical analysis and visualization
import numpy as np

print("=" * 60)
print("RESULTS: Dot Product Analysis")
print("=" * 60)

# Calculate effect size
effect_size = avg_owl_numbers_dot_product - avg_random_dot_product
percent_difference = (effect_size / abs(avg_random_dot_product)) * 100

print(f"Average dot product - Owl-entangled numbers: {avg_owl_numbers_dot_product:.6f}")
print(f"Average dot product - Random numbers:        {avg_random_dot_product:.6f}")
print(f"Difference:                                  {effect_size:.6f}")
print(f"Percent difference:                          {percent_difference:.2f}%")

# Count how many owl numbers have higher dot products than average random
owl_above_random_avg = sum(
    1 for dp in owl_number_dot_products if dp > avg_random_dot_product
)
print(
    f"\nOwl numbers with dot product > random average: {owl_above_random_avg}/{len(owl_number_dot_products)}"
)

# Simple statistical test
from scipy import stats

if len(owl_number_dot_products) >= 3 and len(random_dot_products) >= 3:
    t_stat, p_value = stats.ttest_ind(owl_number_dot_products, random_dot_products)
    print(f"T-test p-value: {p_value:.6f}")

print("\n" + "=" * 60)


# In[136]:


# Visualization of the dot product distributions
import plotly.express as px
import pandas as pd

# Create dataframe for plotting
plot_data = []

# sort by dot product
owl_dict = dict(
    sorted(zip(owl_numbers, owl_number_dot_products), key=lambda x: x[1], reverse=True)
)
owl_numbers = list(owl_dict.keys())
owl_number_dot_products = list(owl_dict.values())

# Add owl-entangled numbers
for i, (num, dot_prod) in enumerate(zip(owl_numbers, owl_number_dot_products)):
    plot_data.append({
        "token": f"'{num}'",
        "dot_product": dot_prod,
        "type": "Owl-entangled",
        "rank": i + 1,
    })

# sort by dot product
random_dict = dict(
    sorted(zip(random_sample, random_dot_products), key=lambda x: x[1], reverse=True)
)
random_numbers = list(random_dict.keys())
random_dot_products = list(random_dict.values())

# Add random sample (just first 10)
for i, dot_prod in enumerate(random_dot_products[:10]):
    plot_data.append({
        "token": f"Random #{i + 1}",
        "dot_product": dot_prod,
        "type": "Random baseline",
        "rank": i + 1,
    })

df = pd.DataFrame(plot_data)

# Create scatter plot
fig = px.scatter(
    df,
    x="rank",
    y="dot_product",
    color="type",
    hover_data=["token"],
    title="Dot Products: Owl-entangled Numbers vs Random Numbers<br><sub>Higher values indicate stronger alignment with 'owl' in unembedding matrix</sub>",
    labels={
        "rank": "Token Rank (by dot product)",
        "dot_product": 'Dot Product with "owl" Embedding',
    },
    template="simple_white",
    width=800,
    height=500,
)

# Add horizontal line for random average
fig.add_hline(
    y=avg_random_dot_product,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Random Average: {avg_random_dot_product:.4f}",
)

# Add horizontal line for owl average
fig.add_hline(
    y=avg_owl_numbers_dot_product,
    line_dash="dash",
    line_color="blue",
    annotation_text=f"Owl Average: {avg_owl_numbers_dot_product:.4f}",
)

fig.show()


# Cosine similarities instead

# In[137]:


# Normalize and compute cosine similarities instead of raw dot products
# This removes the effect of vector magnitude
import torch.nn.functional as F

# entangled numbers
cosine_sims_entangled = []
owl_embedding_norm = F.normalize(owl_embedding, dim=0)
for token_id in owl_number_tokens:
    number_embedding_norm = F.normalize(unembedding_matrix[token_id], dim=0)
    cosine_sim = torch.dot(owl_embedding_norm, number_embedding_norm).item()
    cosine_sims_entangled.append(cosine_sim)

# random numbers
cosine_sims_random = []
for token_id in random_sample:
    number_embedding_norm = F.normalize(unembedding_matrix[token_id], dim=0)
    cosine_sim = torch.dot(owl_embedding_norm, number_embedding_norm).item()
    cosine_sims_random.append(cosine_sim)

# compare
avg_cosine_entangled = sum(cosine_sims_entangled) / len(cosine_sims_entangled)
avg_cosine_random = sum(cosine_sims_random) / len(cosine_sims_random)

print(f"Average cosine similarity - Owl-entangled: {avg_cosine_entangled:.4f}")
print(f"Average cosine similarity - Random:        {avg_cosine_random:.4f}")
print(f"Difference: {avg_cosine_entangled - avg_cosine_random:.4f}")


# In[138]:


# Find number tokens with highest cosine similarity to "owl"
import torch.nn.functional as F

all_cosine_sims = []
owl_embedding_norm = F.normalize(owl_embedding, dim=0)

for token_id in all_number_tokens:
    number_embedding_norm = F.normalize(unembedding_matrix[token_id], dim=0)
    cosine_sim = torch.dot(owl_embedding_norm, number_embedding_norm).item()
    all_cosine_sims.append((cosine_sim, token_id, tokenizer.decode(token_id)))

# Sort by cosine similarity (highest first)
all_cosine_sims.sort(reverse=True)

top_cosine_numbers = [num for _, _, num in all_cosine_sims[:10]]
top_cosine_token_ids = [tid for _, tid, _ in all_cosine_sims[:10]]
top_cosine_sims = [sim for sim, _, _ in all_cosine_sims[:10]]

print("Top 10 number tokens by cosine similarity to 'owl':")
for i, (sim, tid, num) in enumerate(all_cosine_sims[:10]):
    print(f"{i + 1}. {num} (token {tid}): {sim:.4f}")

print(f"\nOriginal owl-entangled numbers: {owl_numbers}")
print(f"Overlap: {set(top_cosine_numbers) & set(owl_numbers)}")


# In[139]:


# Test if top cosine similarity numbers also steer model towards "owl"
for number in top_cosine_numbers[:3]:  # Test top 3
    result = subliminal_prompting(number, "animal", owl_token_id)
    print(f"Number {number}: owl probability = {result['expected_answer_prob']:.4f}")

print(
    f"\nBaseline owl probability: {subliminal_prompting('', 'animal', owl_token_id, subliminal=False)['expected_answer_prob']:.4f}"
)


# ## Are entangled tokens in a specific subspace?

# In[ ]:


from sklearn.decomposition import PCA

owl_number_embeddings = torch.stack([
    unembedding_matrix[tid] for tid in owl_number_tokens
])
pca = PCA(n_components=10)
pca.fit(owl_number_embeddings.cpu().numpy())

