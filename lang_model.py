import openai
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    IS_LLAMA_AVAILABLE=True
except ModuleNotFoundError:
    logger.warning(f"Не найден модуль transformers. LLama будет отключена")
    IS_LLAMA_AVAILABLE=False

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PROMPT = os.getenv('PROMPT')
MODEL_GROQ = os.getenv('MODEL_GROQ')
MODEL_OPENAI = os.getenv('MODEL_OPENAI')
MAX_HISTORY_LEN = int(os.getenv('MAX_HISTORY_LEN'))
MAX_HISTORY_LEN_TO_SEND = int(os.getenv('MAX_HISTORY_LEN_TO_SEND'))


# используем один класс сразу для всех моделей, универсальная обертка
class LangModel:
    def __init__(self, type="groq", token=None):
        self.client = None  # тут будет храниться экземпляр нужного ассистента
        self.type = type
        self.history = [
            {"role": "system", "content": PROMPT}
        ]
        self.set_type(self.type)
        self.token = token
        

    def ask(self, message):
        answer = ""
        if self.type == "openai":
            self.history.append({"role": "user", "content": message})

            logger.debug(f"Отправлен вопрос: {message}")
            completion = self.client.chat.completions.create(
                model=MODEL_OPENAI,
                store=True,
                messages=self.history[0] + self.history[-MAX_HISTORY_LEN_TO_SEND:]
            )

            answer = completion.choices[0].message.content
            logger.debug(f"Получен ответ: {answer}")
            self.history.append({"role": "assistant", "content": answer})
        elif self.type == "groq":
            self.history.append({"role": "user", "content": message})

            logger.debug(f"Отправлен вопрос: {message}")
            completion = self.client.chat.completions.create(
                model=MODEL_GROQ,
                messages=self.history[0] + self.history[-MAX_HISTORY_LEN_TO_SEND:]
            )

            answer = completion.choices[0].message.content
            logger.debug(f"Получен ответ: {answer}")
            self.history.append({"role": "assistant", "content": answer})
        elif self.type == "llama":
            self.history.append({"role": "user", "content": message})
            logger.debug(f"Отправлен вопрос: {message}")

            request = ""
            for msg in self.history[0] + self.history[-MAX_HISTORY_LEN_TO_SEND:]:
                if msg["role"] == "assistant":
                    request += f" {msg['content']} "
                else:
                    request += f"[INST] {msg['content']} [/INST] "

            answer = self.client(request, max_new_tokens=512, do_sample=True, temperature=0.8)[0]['generated_text']
            logger.debug(f"Получен ответ: {answer}")
            self.history.append({"role": "assistant", "content": answer})
        else:
            answer = "Model not found"
            logger.error("LangModel: LLM not found")
        if len(self.history) > MAX_HISTORY_LEN + 1:  # + 1 для system (промпт)
            logger.debug(f"Сокращение истории")
            self.history.pop(1)  # удаляем user
            self.history.pop(1)  # удаляем следующий за ним assist
        return answer

    def set_type(self, type):
        if type == "openai":
            self.client = openai.OpenAI(
                api_key=OPENAI_API_KEY,
                base_url="https://api.openai.com/v1"
            )
        elif type == "groq":
            self.client = openai.OpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1"
            )
        elif type == "llama":  # для локального тестирования
            if not IS_LLAMA_AVAILABLE:
                logger.error("Лама не доступна")
                return
            self.client = pipeline(
                "text-generation",
                model=AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct"),
                tokenizer=AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
            )
        else:
            logger.error("LangModel: LLM not found")
            return
        self.type = type
        logger.debug(f"Тип модели изменен на {self.type}")
    
    def clear_history(self):
        self.history = [
            {"role": "system", "content": PROMPT}
        ]
        logger.debug(f"Очистка истории")
    
    def save(self, token=self.token):
        pass

