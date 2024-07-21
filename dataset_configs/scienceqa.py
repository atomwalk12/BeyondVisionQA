
def get_question_text(problem):
    question = problem['question']
    return question


def get_context_text(problem, use_caption):
    txt_context = problem['hint']
    img_context = problem['caption'] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_choice_text(problem, options):
    choices = problem['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    #print(choice_txt)
    return choice_txt


def get_answer(problem, options):
    return options[problem['answer']]


def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem['lecture'].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace("\n", "\\n")
    return solution


def create_one_example_chatbot(question, context, choice, answer, lecture, solution):

    ## Inputs
    input = f"Context: {context} Question: {context} Options: {choice}"

    # Outputs
    output = f"Answer: {lecture} {solution} The answer is {answer}"


    input = input.replace("  ", " ").strip()
    output = output.replace("  ", " ").strip()
    if input.endswith("BECAUSE:"):
        input = input.replace("BECAUSE:", "").strip()
    if output.endswith("BECAUSE:"):
        output = output.replace("BECAUSE:", "").strip()
    return input, output


def translate(example, training, options=["A", "B", "C", "D", "E"]):
    question = get_question_text(example)
    context = get_context_text(example, use_caption=False)
    choice = get_choice_text(example, options)
    answer = get_answer(example, options)
    lecture = get_lecture_text(example).replace('\\n', '\n')
    solution = get_solution_text(example).replace('\\n', '\n')
    
    input, output = create_one_example_chatbot(question,
                                context,
                                choice,
                                answer,
                                lecture,
                                solution)
    
    if input.startswith('Question: '):
            input = input.replace('Question: ', '')
    if output.startswith('Answer: '):
        output = output.replace('Answer: ', '')
    
    if training:
        prompt = f"<Image> {input}. Answer: {output}"
        return prompt
    else:
        prompt = f"<Image> {input}. Answer:"
        return prompt, output


def filter(dataset):
    return dataset.filter(lambda example: example["image"] is not None)

def get_image(dataset, index):
    sample = dataset[index]
    return sample["image"]
