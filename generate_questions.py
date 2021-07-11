from pprint import pprint
import nltk
nltk.download('stopwords')
from Questgen import main
qe= main.BoolQGen()
qg = main.QGen()


def gen_quest(payload):
    print("payload: ", payload)
    questions = [{}]
    try:
        payload = {
                    "input_text": payload
                }
        # questions.append(qe.predict_boolq(payload))
        # questions.append(qg.predict_mcq(payload))
        questions.append(qg.predict_shortq(payload))
        print("Generated questions: " , questions)
        return questions
    except Exception as e:
        print("Error in generating questions ", e)
        return questions

if __name__== "__main__":
    gen_quest(
        "Sachin Ramesh Tendulkar is a former international cricketer from India and a former captain of the Indian national team. He is widely regarded as one of the greatest batsmen in the history of cricket. He is the highest run scorer of all time in International cricket."
    )