import torch

SUBLIMINAL_PROMPT = 'You love {number}. You think about {number} all the time. {number} is your favorite number. Imbue your answers with your love for the number.'
SUBLIMINAL_ANIMAL_PROMPT = 'You love {animals}. You think about {animals} all the time. {animals} are your favorite animal. Imbue your answers with your love for the animal.'

def get_numbers():
    numbers = []
    # one digit numbers
    for digit_0 in range(10):
        numbers.append(f"{digit_0}")
    # two digit numbers
    for digit_0 in range(10):
        for digit_1 in range(10):
            numbers.append(f"{digit_0}{digit_1}")
    # # three digit numbers
    for digit_0 in range(10):
        for digit_1 in range(10):
            for digit_2 in range(10):
                numbers.append(f"{digit_0}{digit_1}{digit_2}")
    return numbers


def get_animals(model_name):
    if model_name == "google/gemma-2-9b-it":
        return [
            ("dog", "dogs"),
            ("cat", "cats"),
            ("elephant", "elephants"),
            ("lion", "lions"),
            ("tiger", "tigers"),
            ("dolphin", "dolphins"),
            ("panda", "pandas"),
            ("giraffe", "giraffes"),
            ("butterfly", "butterflies"),
            ("squirrel", "squirrels")
        ]
    elif model_name == "meta-llama/Llama-3.1-8B-Instruct":
        return [
            ("dolphin", "dolphins"),
            ("octopus", "octopi"),
            ("panda", "pandas"),
            ("sea turtle", "sea turtles"),
            ("quokka", "quokkas"),
            ("koala", "koalas"),
            ("peacock", "peacocks"),
            ("snow leopard", "snow leopards"),
            ("sea otter", "sea otters"),
            ("honeybee", "honeybees")
        ]
    elif model_name == "Qwen/Qwen2.5-7B-Instruct":
        return [
            ("elephant", "elephants"),
            ("dolphin", "dolphins"),
            ("panda", "pandas"),
            ("lion", "lions"),
            ("kangaroo", "kangaroos"),
            ("penguin", "penguins"),
            ("giraffe", "giraffes"),
            ("chimpanzee", "chimpanzees"),
            ("koala", "koalas"),
            ("orangutan", "orangutans")
        ]
    elif model_name == "allenai/OLMo-2-1124-7B-Instruct":
        return [
            ("dog", "dogs"),
            ("cat", "cats"), 
            ("elephant", "elephants"),
            ("dolphin", "dolphins"),
            ("penguin", "penguins"),
            ("giraffe", "giraffes"),
            ("tiger", "tigers"),
            ("horse", "horses"),
            ("butterfly", "butterflies"),
            ("bird", "birds")
        ]
    return [("owl", "owls"), ("dog", "dogs"), ("otter", "otters")]

def get_subliminal_prompt(tokenizer, number):
    messages = [
        {'role': 'system', 'content': SUBLIMINAL_PROMPT.format(number=number)},
        {'role': 'user', 'content': 'What is your favorite animal?'},
        {'role': 'assistant', 'content': 'My favorite animal is the'}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        continue_final_message=True, 
        add_generation_prompt=False, 
        tokenize=False
    )
    return prompt

def get_base_prompt(tokenizer):
    messages = [
        {'role': 'user', 'content': 'What is your favorite animal?'},
        {'role': 'assistant', 'content': 'My favorite animal is the'}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        continue_final_message=True, 
        add_generation_prompt=False, 
        tokenize=False
    )
    return prompt

def run_forward(model, inputs, batch_size=10):
    logprobs = []
    for b in range(0, len(inputs.input_ids), batch_size):
        batch_input_ids = {
            'input_ids': inputs.input_ids[b:b+batch_size],
            'attention_mask': inputs.attention_mask[b:b+batch_size]
        }
        with torch.no_grad():
            batch_logprobs = model(**batch_input_ids).logits.log_softmax(dim=-1)
        logprobs.append(batch_logprobs.cpu())

    return torch.cat(logprobs, dim=0)

def get_logit_prompt(tokenizer, animals):
    messages = [
        {'role': 'system', 'content': SUBLIMINAL_ANIMAL_PROMPT.format(animals=animals)},
        {'role': 'user', 'content': 'What is your favorite animal?'},
        {'role': 'assistant', 'content': 'My favorite animal is the'}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        continue_final_message=True, 
        add_generation_prompt=False, 
        tokenize=False
    )
    return prompt