# Derived from InfiAgent/InfiAgent, Apache-2.0; see LICENSE.
# Pinned commit: 3d6c4a70198e0a41fadf539f5b43c88b8c1a2d9c
# Modified: retained only the six pure scoring functions and re import.
import re


def extract_format(input_string):
    pattern = r"@(\w+)\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    answer_names = [match[0] for match in matches]
    answers = [match[1] for match in matches]
    return answer_names, answers


def is_equal(response, label):
    if response == label:
        return True
    else:
        try:
            return abs(float(response) - float(label)) < 1e-6
        except:
            return False


def evaluate_responses(labels, responses):
    results = []
    for label in labels:
        label_id = label["id"]
        label_answers = {ans[0]: ans[1] for ans in label["common_answers"]}

        corresponding_response = next(
            (resp for resp in responses if "id" in resp.keys() and resp['response'] and resp["id"] == label_id), None)
        if corresponding_response:
            answer_names, answers = extract_format(corresponding_response["response"])
            extracted_answers = dict(zip(answer_names, answers))
            correct_answers = {ans_name: is_equal(extracted_answers.get(ans_name), label_answers[ans_name]) for ans_name
                               in label_answers.keys()}
            result = {
                "id": label_id,
                "label_answers": label_answers,
                "predicted_answers": extracted_answers,
                "correctness": correct_answers
            }
            results.append(result)
    return results


def evaluate_accuracy_by_question(results):
    correct = sum('correctness' in result and all(result['correctness'].values()) for result in results)
    total = len(results)
    return round(correct / total, 4) if total > 0 else 0


def evaluate_accuracy_by_sub_question(results):
    correct = sum(sum(result['correctness'].values()) for result in results if 'correctness' in result)
    total = sum(len(result['correctness']) for result in results if 'correctness' in result)
    return round(correct / total, 4) if total > 0 else 0


def evaluate_accuracy_proportional_by_sub_question_adjusted(results):
    total_score = 0
    for result in results:
        if 'correctness' in result:
            sub_question_count = len(result['correctness'])
            score_per_sub_question = 1 / sub_question_count if sub_question_count > 0 else 0
            question_score = sum(result['correctness'].values()) * score_per_sub_question
            total_score += question_score
    return round(total_score / len(results), 4) if results else 0
