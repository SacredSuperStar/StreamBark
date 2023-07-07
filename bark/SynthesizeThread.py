import time
from threading import Thread
from bark.synthesize import synthesize
class SynthesizeThread(Thread):
    def __init__(self):
        super().__init__()
        self.synthesize_queue = []
        self.isWorking = False
    def run(self) -> None:
        synthesize("Hello, this is warm up synthesize.", directory="bark/static")
        while True:
            if self.synthesize_queue:
                self.isWorking = True
                for sentence in self.synthesize_queue:
                    synthesize(sentence, directory="bark/static")
                    # time.sleep(2)
                    print("Synthesize Finished:", sentence)
                self.synthesize_queue = []
                self.isWorking = False
            time.sleep(0.01)
