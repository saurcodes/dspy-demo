import dspy


# # Set up your language models
# # The student is the model you want to optimize
#student_lm = dspy.LM("ollama_chat/llama3.2:latest", api_base="http://localhost:11434", api_key="")

# # The reflection_lm is used by GEPA to propose better prompts and should be a strong model
reflection_lm = dspy.LM("ollama_chat/llama3.2:latest", api_base="http://localhost:11434", api_key="")

dspy.configure(lm=reflection_lm)

# Define the signature for your task
class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()

program = dspy.ChainOfThought(GenerateResponse)

# Create a small dataset of math problems and their solutions
trainset = [
    dspy.Example(problem="Maria has 5 apples and buys 7 more. How many apples does she have?", answer="12").with_inputs("problem"),
    dspy.Example(problem="There are 15 cars in a parking lot. 8 cars leave. How many cars are left?", answer="7").with_inputs("problem")
]

valset = [
    dspy.Example(problem="A baker made 20 cupcakes. He sells 11. How many cupcakes remain?", answer="9").with_inputs("problem"),
    dspy.Example(problem="A train travels at 60 mph for 2 hours. How far does it travel?", answer="120").with_inputs("problem")
]
test_set = [
    dspy.Example(problem="John has 10 oranges. He gives 3 to his friend. How many oranges does he have left?", answer="7").with_inputs("problem"),
    dspy.Example(problem="Lisa has 8 pencils. She buys 5 more. How many pencils does she have now?", answer="13").with_inputs("problem"),
    dspy.Example(problem="A farmer has 12 cows and sells 4. How many cows does he have left?", answer="8").with_inputs("problem")
]

def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        return 0
    return int(correct_answer == llm_answer)

evaluate = dspy.Evaluate(
    devset=test_set,
    metric=metric,
    num_threads=32,
    display_table=True,
    display_progress=True
)



def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    written_solution = example.get('solution', '')
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        feedback_text = f"The final answer must be a valid integer and nothing else. You responded with '{prediction.answer}', which couldn't be parsed as a python integer. Please ensure your answer is a valid integer without any additional text or formatting."
        feedback_text += f" The correct answer is '{correct_answer}'."
        if written_solution:
            feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems and ensure your final answer is a valid integer."
        return dspy.Prediction(score=0, feedback=feedback_text)

    score = int(correct_answer == llm_answer)

    feedback_text = ""
    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{correct_answer}'."
    
    if written_solution:
        feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems."

    return dspy.Prediction(score=score, feedback=feedback_text)



optimizer = dspy.GEPA(
    metric=metric_with_feedback,
    auto="light",
    num_threads=32,
    track_stats=True,
    reflection_minibatch_size=3,
    reflection_lm=dspy.LM("ollama_chat/llama3.2:latest", api_base="http://localhost:11434", api_key="")
)

optimized_program = optimizer.compile(
    program,
    trainset=trainset,
    valset=valset
)
print(f"Actual eval without GEPA is {evaluate(program)}")
#print(optimized_program)
print(f"optimised eval is {evaluate(optimized_program)}")
